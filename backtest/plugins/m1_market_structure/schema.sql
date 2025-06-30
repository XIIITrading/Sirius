-- backtest/plugins/m1_market_structure/schema.sql
-- Table schema for M1 Market Structure calculation results

CREATE TABLE public.bt_m1_market_structure (
  uid character varying(50) not null,
  signal_direction character varying(20) null,
  signal_strength numeric(5, 2) null,
  signal_confidence numeric(5, 2) null,
  structure_type character varying(20) null, -- BOS, CHoCH, or NONE
  current_trend character varying(20) null,   -- BULL, BEAR, or NEUTRAL
  last_high_fractal numeric(10, 4) null,
  last_low_fractal numeric(10, 4) null,
  last_break_type character varying(20) null,
  last_break_time timestamp with time zone null,
  last_break_price numeric(10, 4) null,
  fractal_count integer null,
  structure_breaks integer null,
  trend_changes integer null,
  candles_processed integer null,
  reason text null,
  constraint bt_m1_market_structure_pkey primary key (uid),
  constraint bt_m1_market_structure_uid_fkey foreign key (uid) references bt_index (uid) on delete cascade
) tablespace pg_default;

-- Create index for faster queries on structure type
CREATE INDEX idx_bt_m1_market_structure_type ON public.bt_m1_market_structure(structure_type);

-- Create index for trend analysis
CREATE INDEX idx_bt_m1_market_structure_trend ON public.bt_m1_market_structure(current_trend);