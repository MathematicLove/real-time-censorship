import os
import queue
import threading
import time
from datetime import datetime, timezone
from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy import create_engine, func, insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from app import config, APP_VERSION
from app.logger import log

def utcnow():
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass

class Run(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    started_at: Mapped[object] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    device: Mapped[str] = mapped_column(String(16), nullable=False)
    providers: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[str] = mapped_column(String(16), nullable=False)

class Detection(Base):
    __tablename__ = "detections"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    session_id: Mapped[str] = mapped_column(String(32), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    person_no: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str] = mapped_column(String(48), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    box_x: Mapped[int] = mapped_column(Integer, nullable=False)
    box_y: Mapped[int] = mapped_column(Integer, nullable=False)
    box_w: Mapped[int] = mapped_column(Integer, nullable=False)
    box_h: Mapped[int] = mapped_column(Integer, nullable=False)
    blurred: Mapped[bool] = mapped_column(Boolean, nullable=False)
    __table_args__ = (
        Index("detections_created_at_idx", created_at.desc()),
        Index("detections_session_idx", session_id),
        Index("detections_person_idx", person_no),
        Index("detections_label_idx", label),
    )

class Database:
    def __init__(self, url=None):
        self.url = make_url(url or config.DB_URL)
        self.backend = self.url.get_backend_name()
        self.engine = None
        self.factory = None
        self.queue = queue.Queue(maxsize=20000)
        self.worker = None
        self.running = False
        self.ready = False
        self.dropped = 0
        self.written = 0

    def label(self):
        if self.backend == "sqlite":
            return "sqlite " + str(self.url.database)
        return self.backend + " " + str(self.url.database) + " on " + str(self.url.host) + " as " + str(self.url.username)

    def options(self):
        if self.backend == "sqlite":
            return {"connect_args": {"check_same_thread": False, "timeout": 30}, "pool_pre_ping": True}
        return {"pool_pre_ping": True, "pool_size": 5, "max_overflow": 5, "pool_recycle": 1800}

    def connect(self, device="cpu", providers=None):
        attempts = max(1, config.DB_CONNECT_RETRIES) if self.backend != "sqlite" else 1
        for attempt in range(1, attempts + 1):
            try:
                if config.DB_AUTO_CREATE:
                    self.ensure_database()
                self.engine = create_engine(self.url, **self.options())
                self.tune()
                Base.metadata.create_all(self.engine)
                self.factory = sessionmaker(self.engine, expire_on_commit=False)
                self.register(device, providers or [])
                self.ready = True
                self.running = True
                self.worker = threading.Thread(target=self._loop, daemon=True)
                self.worker.start()
                log("database ready " + self.label())
                return True
            except Exception as error:
                reason = str(error).replace("\n", " ")[:200]
                if attempt < attempts:
                    log("database attempt " + str(attempt) + " of " + str(attempts) + " failed, retry in " + str(config.DB_CONNECT_DELAY) + "s, reason " + reason)
                    time.sleep(config.DB_CONNECT_DELAY)
                else:
                    log("database unavailable, logging to file only, reason " + reason)
        self.ready = False
        return False

    def ensure_database(self):
        if self.backend == "sqlite":
            path = self.url.database
            if path and path != ":memory:":
                folder = os.path.dirname(os.path.abspath(path))
                if folder:
                    os.makedirs(folder, exist_ok=True)
            return
        if self.backend != "postgresql":
            return
        name = self.url.database
        admin = create_engine(self.url.set(database=config.DB_MAINTENANCE), isolation_level="AUTOCOMMIT", pool_pre_ping=True)
        try:
            with admin.connect() as connection:
                found = connection.execute(select(text("1")).select_from(text("pg_database")).where(text("datname = :name")), {"name": name}).scalar()
                if not found:
                    connection.execute(text('create database "' + name.replace('"', '""') + '"'))
                    log("database " + name + " created")
        finally:
            admin.dispose()

    def tune(self):
        if self.backend != "sqlite":
            return
        with self.engine.connect() as connection:
            connection.execute(text("pragma journal_mode=wal"))
            connection.execute(text("pragma synchronous=normal"))
            connection.commit()

    def register(self, device, providers):
        with self.factory() as session:
            if session.get(Run, config.SESSION_ID) is None:
                session.add(Run(
                    id=config.SESSION_ID,
                    device=str(device),
                    providers=",".join(providers)[:256],
                    version=APP_VERSION,
                ))
                session.commit()

    def record(self, source, person_no, label, score, box, blurred):
        if not self.ready:
            return False
        item = {
            "created_at": utcnow(),
            "session_id": config.SESSION_ID,
            "source": str(source),
            "person_no": int(person_no),
            "label": str(label),
            "score": float(score),
            "box_x": int(box[0]),
            "box_y": int(box[1]),
            "box_w": int(box[2]),
            "box_h": int(box[3]),
            "blurred": bool(blurred),
        }
        try:
            self.queue.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _loop(self):
        while self.running:
            try:
                first = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < 200:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            self._flush(batch)

    def _flush(self, batch):
        try:
            with self.factory() as session:
                session.execute(insert(Detection), batch)
                session.commit()
            self.written += len(batch)
        except SQLAlchemyError as error:
            log("database write failed for " + str(len(batch)) + " rows, reason " + str(error).replace("\n", " ")[:200])

    def recent(self, limit=100, source=None, person_no=None, blurred=None, session_id=None):
        if not self.ready:
            return []
        statement = select(Detection).order_by(Detection.created_at.desc(), Detection.id.desc()).limit(int(limit))
        if source is not None:
            statement = statement.where(Detection.source == source)
        if person_no is not None:
            statement = statement.where(Detection.person_no == int(person_no))
        if blurred is not None:
            statement = statement.where(Detection.blurred == bool(blurred))
        if session_id is not None:
            statement = statement.where(Detection.session_id == session_id)
        with self.factory() as session:
            rows = session.execute(statement).scalars().all()
        return [self.as_dict(row) for row in rows]

    def as_dict(self, row):
        stamp = row.created_at
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return {
            "id": row.id,
            "created_at": stamp.isoformat(),
            "session_id": row.session_id,
            "source": row.source,
            "person_no": row.person_no,
            "label": row.label,
            "score": round(float(row.score), 4),
            "box_x": row.box_x,
            "box_y": row.box_y,
            "box_w": row.box_w,
            "box_h": row.box_h,
            "blurred": bool(row.blurred),
        }

    def stats(self):
        if not self.ready:
            return {"available": False, "backend": self.backend, "total": 0, "blurred": 0, "persons": 0, "sessions": 0, "by_label": {}}
        with self.factory() as session:
            total = session.execute(select(func.count()).select_from(Detection)).scalar() or 0
            blurred = session.execute(select(func.count()).select_from(Detection).where(Detection.blurred.is_(True))).scalar() or 0
            persons = session.execute(select(func.count(func.distinct(Detection.person_no)))).scalar() or 0
            runs = session.execute(select(func.count()).select_from(Run)).scalar() or 0
            labels = session.execute(select(Detection.label, func.count()).group_by(Detection.label).order_by(func.count().desc())).all()
        return {
            "available": True,
            "backend": self.backend,
            "total": int(total),
            "blurred": int(blurred),
            "persons": int(persons),
            "sessions": int(runs),
            "by_label": {name: int(count) for name, count in labels},
            "queued": self.queue.qsize(),
            "written": self.written,
            "dropped": self.dropped,
        }

    def close(self):
        self.running = False
        if self.worker is not None:
            self.worker.join(timeout=2.0)
        pending = []
        while True:
            try:
                pending.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if pending and self.ready:
            self._flush(pending)
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None
        self.factory = None
        self.ready = False