Idea generation and general framework for the Daily Playbook
Location: market_review\pre_market\daily_playbook

The purpose of this Daily Playbook is to enter the overall plan for the day based on pre-market analysis of zones, levels, and pivots
This Daily Playbook will serve three purposes:
    1. Provide a clean plan at the start of the day so that zones, levels, and pivots can be monitored for entry
    2. Serve as the basis for a user-generated scanner which will notify the trader when the price has enetered, and then either held to the zone or broken the zone.
    3. Provide the general framework for future automations based on the data collected and obervations of consistent zone, level, pivot selections over time.

Module 1: Data Entry
Location: market_review\pre_market\daily_playbook\data
- Data will be entered into Excel in the below format
- Mornign images of the zones will be provided in .png format and saved into a Dropbox with the Excel
- Python tool will need to be created called playbook_save.py
    - Location: market_review\pre_market\daily_playbook\database\playbook_save
    - This playbook save will run the workflow to ingest into Supabase, and should request the Dropbox file path location that will be copied and pasted into the terminal.
    - The file names will be the same each and every day so that a PDF repost can be generated to provide to my students
        - Location: market_review\pre_market\daily_playbook\database\playbook_report
    - That PDF file will be saved in the same Dropbox location documenting my morning playbook.

# Trading Plays Example Data

| Symbol | Price Low | Price High | Bullish Target | Bearish Target | Bullish Statement | Bearish Statement | Rank |
|--------|-----------|------------|----------------|----------------|-------------------|-------------------|------|
| ADSK | 293.28 | 297.88 | 310.00 | 286.00 | Hold above 297.88, then enter bullish for target 310.00 | Break below 293.28, then enter bearish for target 286.00 | primary |
| TSLA | 330.00 | 320.00 | 338.00 | 314.00 | Hold above 300, then enter bullish for target 338.00 | Hold below 320.00, then enter bearish for target 314 (partial) then 305 | primary |
| MU | 122.00 | 123.50 | 124.50 | 116.00 | Hold above 123.50, enter bullish for target 124.50 | Break below 122.00, enter bearish for target 116.00 | primary |
| MU | 115.93 | 117.12 | 119.59 | 114.18 | Hold above 117.12, enter bullish for reversal target 119.59 | Break below 115.93, enter bearish for target 114.18 | secondary |
| MU | 122.03 | 123.50 | 124.44 | 120.86 | Break above 123.50, enter bullish for target 124.44 | Hold below 122.03, enter bearish for target 120.86 | secondary |
| NVDA | 164.87 | 165.18 | 166.00 | 164.08 | Hold above 165.18, enter bullish for target 166.00 | Price break below 164.87, enter bearish for target 164.08 | primary |
| NVDA | 163.67 | 164.05 | 164.88 | 162.63 | Hold above 164.05, enter bullish for target 164.885 | Break below 163.67, enter bearish for target 162.63 | secondary |
| NVDA | 166.02 | 166.47 | 167.26 | 165.17 | Break above 166.47, enter bullish for target 167.27 | Holds below 166.02, enter bearish for target 165.17 | secondary |

 This data will be uploaded to Supabase to serve two primary functions:
    1. Trading Day System
    2. Post-Market Analysis

Trading Day System
Location: market_review\pre_market\daily_playbook\calculations
    1. Serve as the basis for calculation and filtering of the plays
        - Filter 1: R:R will be calculated with the assumption that the stop will be placed on the opposite side of the zone. 
            - Taking the stop distance from entry and comparing it to the target price relative to the entry - the Risk / Reward Ratio can be calculated
                - Bullish Trade Direction = (bullish_target - price_high) / (price_high - price_low)
                - Bearish Trade Direction = (price_low - bearish_target) / (price_high - price_low)
    2. Provide simple inputs to the Daily Playbook Scanner (built in PyQT6 with Polygon Data)
        - The Daily Playbook Scanner will be a grid style scanner that will display the playbook tickers in seperate grids based on the tickers that are entered.
        - 1-Minute close price will be monitored using the Polygon Websocket for relative proximity to the zones, or if the price closes within the zone.
        - If the price closed within the zone, then the zone will be highlighted and an alert will display indicating that this play is active given follow through.
        - The follow through will be if the price closes outside of the zone either in a hold and rejection pattern, or a break pattern. 
        - Once the price closes outside of the zone an alert will be provided that the entry criteria should be viewed in the Live Monitor Tool for possible entry

Post-Market Analys
<Section Coming Soon>

Supabase Schema
-- Main daily_playbook table
CREATE TABLE daily_playbook (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    price_low DECIMAL(10,2),
    price_high DECIMAL(10,2),
    bullish_target DECIMAL(10,2),
    bearish_target DECIMAL(10,2),
    bullish_statement TEXT,
    bearish_statement TEXT,
    rank VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

