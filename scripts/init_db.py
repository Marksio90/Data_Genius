"""
DataGenius PRO - Initialize Database
Script to initialize database with tables
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

from config.settings import settings
from db.models import Base


def init_database(drop_existing: bool = False):
    """
    Initialize database with tables
    
    Args:
        drop_existing: Drop existing tables before creating new ones
    """
    
    logger.info("Initializing database...")
    
    try:
        # Create engine
        database_url = settings.get_database_url()
        logger.info(f"Connecting to database: {database_url}")
        
        engine = create_engine(database_url, echo=settings.ENABLE_SQL_ECHO)
        
        # Drop existing tables if requested
        if drop_existing:
            logger.warning("Dropping existing tables...")
            Base.metadata.drop_all(engine)
            logger.info("Existing tables dropped")
        
        # Create all tables
        logger.info("Creating tables...")
        Base.metadata.create_all(engine)
        
        # Verify tables
        inspector = engine.inspect()
        tables = inspector.get_table_names()
        
        logger.success(f"Database initialized successfully!")
        logger.info(f"Created tables: {', '.join(tables)}")
        
        # Create session to test connection
        Session = sessionmaker(bind=engine)
        session = Session()
        session.close()
        
        logger.success("Database connection tested successfully!")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def seed_sample_data(session):
    """
    Seed database with sample data (optional)
    
    Args:
        session: Database session
    """
    
    from db.models import Session as SessionModel
    from core.utils import generate_session_id
    
    logger.info("Seeding sample data...")
    
    try:
        # Create sample session
        sample_session = SessionModel(
            session_id=generate_session_id(),
            status="completed",
            pipeline_stage="training_complete",
            n_rows=150,
            n_columns=5,
            target_column="species",
            problem_type="classification"
        )
        
        session.add(sample_session)
        session.commit()
        
        logger.success("Sample data seeded successfully!")
    
    except Exception as e:
        logger.error(f"Failed to seed sample data: {e}")
        session.rollback()


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize DataGenius PRO database")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing tables before creating new ones"
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed database with sample data"
    )
    
    args = parser.parse_args()
    
    # Initialize database
    success = init_database(drop_existing=args.drop)
    
    if not success:
        logger.error("Database initialization failed")
        sys.exit(1)
    
    # Seed sample data if requested
    if args.seed:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        engine = create_engine(settings.get_database_url())
        Session = sessionmaker(bind=engine)
        session = Session()
        
        seed_sample_data(session)
        
        session.close()
    
    logger.success("✅ Database setup complete!")
    print("\n" + "="*50)
    print("📊 Database initialized successfully!")
    print("="*50)
    print(f"\nDatabase URL: {settings.get_database_url()}")
    print("\nYou can now run the application:")
    print("  streamlit run app.py")
    print("\nOr use Make:")
    print("  make run")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()