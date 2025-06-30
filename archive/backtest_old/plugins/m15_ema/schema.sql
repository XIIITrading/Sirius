-- backtest/plugins/m15_ema/schema.sql
-- Table schema for M15 EMA calculation results

CREATE TABLE public.bt_m15_ema (
  uid character varying(50) not null,
  signal_direction character varying(20) null,
  signal_strength numeric(5, 2) null,
  signal_confidence numeric(5, 2) null,
  ema_9 numeric(10, 4) null,
  ema_21 numeric(10, 4) null,
  ema_spread numeric(10, 4) null,
  ema_spread_pct numeric(5, 2) null,
  price_vs_ema9 character varying(20) null,
  trend_strength numeric(5, 2) null,
  reason text null,
  constraint bt_m15_ema_pkey primary key (uid),
  constraint bt_m15_ema_uid_fkey foreign key (uid) references bt_index (uid) on delete cascade
) tablespace pg_default;