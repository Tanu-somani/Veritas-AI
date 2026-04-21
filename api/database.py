import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

DATABASE_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/fakenewsdb")

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class PredictionLog(Base):
        __tablename__ = "prediction_logs"

        id = Column(Integer, primary_key=True, index=True)
        text_snippet = Column(String(500))
        predicted_label = Column(String(50))
        confidence = Column(Float)
        explanation = Column(String)
        timestamp = Column(DateTime, default=datetime.utcnow)

    # Create tables
    Base.metadata.create_all(bind=engine)
    db_available = True
except Exception as e:
    print(f"Warning: Database connection failed. {e}")
    db_available = False

def log_prediction(text: str, label: str, confidence: float, explanation: str):
    if not db_available:
        return
    
    try:
        db = SessionLocal()
        # Only log a snippet to save space
        snippet = text[:497] + "..." if len(text) > 500 else text
        db_log = PredictionLog(
            text_snippet=snippet,
            predicted_label=label,
            confidence=confidence,
            explanation=explanation
        )
        db.add(db_log)
        db.commit()
        db.close()
    except Exception as e:
        print(f"Failed to log prediction to database: {e}")
