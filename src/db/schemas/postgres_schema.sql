CREATE TABLE IF NOT EXISTS hotel_reviews (
    hotel_name       TEXT,
    city             TEXT,
    province         TEXT,
    review_date      TIMESTAMP,
    rating           DOUBLE PRECISION,
    review_full_text TEXT,
    sentiment        INTEGER,
    region           TEXT,
    division         TEXT
);