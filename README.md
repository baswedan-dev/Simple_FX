# Simplified FX Trading System

**Version:** 1.0  
**Target Performance:** 60-62% Accuracy on Filtered Signals  
**Philosophy:** Simplicity • Robustness • Maintainability

---

## 📋 Executive Summary

A production-grade forex trading system designed for sustainable 60% directional accuracy over years, not months. Built on the principle that **fewer components = fewer failure points**.

### Target Performance Metrics
- ✅ 60-62% accuracy on filtered signals
- ✅ 15-25% trade frequency (selective execution)
- ✅ Sharpe ratio > 1.5
- ✅ Maximum drawdown < 15%
- ✅ System uptime > 99%

---

## 🏗️ System Architecture

### 6-Layer Design

```
Layer 1: Data Pipeline → Clean OHLC, single source, validation
         ↓
Layer 2: Non-Repainting Features → 8-12 causal indicators only
         ↓
Layer 3: Simple Regime Detection → TREND/RANGE via volatility + slope
         ↓
Layer 4: Single Gradient Booster → XGBoost with regime as feature
         ↓
Layer 5: Execution Gate → Probability + Regime + Risk filters
         ↓
Layer 6: Risk Management → ATR-based stops, fixed % risk
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Polygon.io API key (free tier supported)
- 28 currency pairs available

### Installation

```bash
# 1. Clone the repository
cd ~/Documents/Simple_FX

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# 5. Set up environment variable
export POLYGON_API_KEY="your_api_key_here"

# 6. Create required directories
mkdir -p data/raw logs
```

### Configuration

Edit `config/config.yml` to customize:
- Data source settings
- Cache TTL (default: 24 hours)
- Validation parameters
- Feature groups
- Model hyperparameters

Edit `config/pairs.yml` to select currency pairs (7 working pairs included by default)

---

## 📊 Current Status: Phase 1 Complete

### ✅ Phase 1: Data Pipeline (STABILIZATION PERIOD)

**Status:** In 2-3 day stabilization period

**Deliverables:**
- [x] Polygon.io API integration
- [x] OHLC validation suite
- [x] Local cache implementation
- [x] Unit tests for data integrity (20 tests passing)

**Success Criteria Met:**
- ✅ Pipeline runs daily without errors
- ✅ All validation checks pass (20/20 tests)
- ✅ Cache hit rate > 90%

**Daily Operations:**
```bash
# Run data pipeline (execute daily)
python run_phase1_data.py

# Run validation tests (execute daily)
pytest tests/test_data/ -v
```

**Current Data:**
- 7 currency pairs: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, EUR/JPY, GBP/JPY
- 623 records per pair (~2 years of daily data)
- Date range: 2024-01-19 to 2026-01-16

---

## 🗂️ Project Structure

```
FX_Daily_Bias_Simplified/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation config
├── config/
│   ├── config.yml                    # Master configuration
│   └── pairs.yml                     # Currency pairs list
├── src/
│   ├── data/                         # Layer 1: Data Pipeline
│   │   ├── ingestion.py              # Main data ingestion
│   │   ├── validation.py             # OHLC validation
│   │   ├── cache.py                  # Local caching system
│   │   └── polygon_client.py         # Polygon.io API client
│   └── utils/
│       ├── logger.py                 # Logging utilities
│       └── config_loader.py          # Configuration loader
├── tests/
│   ├── conftest.py                   # Pytest fixtures
│   └── test_data/                    # Phase 1 tests
│       ├── test_cache.py             # Cache tests (6 tests)
│       ├── test_ingestion.py         # Ingestion tests (7 tests)
│       └── test_validation.py        # Validation tests (7 tests)
├── scripts/
│   ├── run_phase1_data.py            # Phase 1 execution script
│   ├── test_available_pairs.py       # Test pair availability
│   └── diagnose_historical_data.py   # Data diagnostic tool
├── data/
│   └── raw/                          # Cached OHLC data
└── logs/                             # Application logs
```

---

## 🧪 Testing

### Run All Phase 1 Tests
```bash
pytest tests/test_data/ -v
```

### Test Specific Components
```bash
# Cache tests only
pytest tests/test_data/test_cache.py -v

# Validation tests only
pytest tests/test_data/test_validation.py -v

# Ingestion tests only
pytest tests/test_data/test_ingestion.py -v
```

### Test Coverage
```bash
pytest tests/test_data/ --cov=src.data --cov-report=html
```

---

## 🛠️ Utility Scripts

### Test Currency Pair Availability
```bash
python scripts/test_available_pairs.py
```
Tests which forex pairs are available on your Polygon.io subscription.

### Diagnose Historical Data
```bash
python scripts/diagnose_historical_data.py
```
Checks historical data availability across different time periods (1 month to 3 years).

---

## 📈 Data Pipeline Details

### Data Source
- **Provider:** Polygon.io
- **Timeframe:** Daily (D1)
- **History:** 2 years (730 days)
- **Update Frequency:** Daily

### Validation Checks
All data must pass these checks:
1. ✅ No null values
2. ✅ OHLC logic (high ≥ low)
3. ✅ Close within high-low range
4. ✅ No excessive gaps (> 5%)
5. ✅ Timestamps sorted
6. ✅ No duplicate timestamps
7. ✅ Sufficient data (≥ 30 days)

### Cache System
- **Location:** `data/raw/`
- **Format:** CSV files
- **TTL:** 24 hours
- **Naming:** `{PAIR}_daily.csv` (e.g., `EUR_USD_daily.csv`)

---

## 🔄 Implementation Roadmap

| Phase | Duration | Status | Deliverables |
|-------|----------|--------|--------------|
| **Phase 1: Data** | Week 1-2 | 🟡 Stabilization | Data pipeline, validation, cache, tests |
| **Phase 2: Features** | Week 3-4 | ⏸️ Pending | 8-12 non-repainting features, anti-leakage tests |
| **Phase 3: Regime** | Week 5 | ⏸️ Pending | Rule-based regime detector |
| **Phase 4: Model** | Week 6-7 | ⏸️ Pending | XGBoost, walk-forward validation |
| **Phase 5: Execution** | Week 8 | ⏸️ Pending | Execution gate, signal generation |
| **Phase 6: Risk** | Week 9 | ⏸️ Pending | Position sizing, stops, limits |
| **Phase 7: Paper Trading** | Week 10-13 | ⏸️ Pending | 4-week validation period |

---

## ⚠️ Critical Rules

### Data Pipeline (Phase 1)
- ✅ One timestamp = one source (no mixing)
- ✅ Forward-fill gaps max 2 bars only
- ✅ Never backfill (look-ahead risk)
- ✅ Log every data issue with timestamp
- ✅ Daily integrity check before each run

### Anti-Leakage (All Phases)
- ❌ Never use `.pct_change()` without `.shift(1)`
- ❌ Never use centered windows (`center=True`)
- ❌ Never use future data in normalization
- ❌ Never fit scalers before train/test split
- ❌ Never use test set for hyperparameters

---

## 📝 Daily Checklist (Phase 1 Stabilization)

Run these commands daily for 2-3 days:

```bash
# 1. Fetch latest data
python run_phase1_data.py

# 2. Validate all tests pass
pytest tests/test_data/ -v

# 3. Check logs
tail -f logs/phase1_*.log
```

**Monitor for:**
- ✅ New data fetched successfully
- ✅ Cache hits on subsequent runs
- ✅ All 20 tests passing
- ✅ Record counts incrementing by ~1 per day
- ✅ No errors or warnings

---

## 🎯 Success Criteria

### Phase 1 (Current)
- [x] Pipeline runs daily without errors
- [x] All validation checks pass
- [x] Cache hit rate > 90%
- [ ] **2-3 days of stable operation** ← IN PROGRESS

### Phase 2 (Next)
- [ ] All features pass causality tests
- [ ] Shuffle test passes
- [ ] No null/inf values
- [ ] Feature correlations < 0.85

---

## 📚 Documentation

- **Architecture:** See `Simple_FX_Trading_System.pdf`
- **Working Prompt:** See `Working_Prompt.pdf`
- **Code Documentation:** Inline docstrings (Google style)

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Reinstall in development mode
pip uninstall simple-fx -y
pip install -e .
```

### API Errors
```bash
# Test API key
echo $POLYGON_API_KEY

# Test pair availability
python scripts/test_available_pairs.py
```

### Cache Issues
```bash
# Clear cache
rm -rf data/raw/*.csv

# Rebuild cache
python run_phase1_data.py
```

---

## 📊 Performance Expectations

### Phase 1 (Data)
- **Latency:** < 1 second (cache hit)
- **Latency:** 10-15 seconds per pair (API fetch)
- **Cache Hit Rate:** > 90%
- **Data Quality:** 100% validation pass rate

### Final System (Phase 7)
- **Raw Model Accuracy:** 52-56%
- **After Regime Filtering:** 58-62%
- **After Execution Gate:** 60-65%
- **Trade Frequency:** 15-25% of trading days
- **Sharpe Ratio:** > 1.5
- **Max Drawdown:** < 15%

---

## 🤝 Contributing

This is a personal trading system. No external contributions accepted.

---

## ⚖️ License

Proprietary. For personal use only.

---

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review error messages carefully
3. Consult architecture PDF
4. Run diagnostic scripts

---

## 🔐 Security Notes

- **Never commit API keys** to version control
- Use environment variables for sensitive data
- Keep `config/config.yml` out of public repositories
- API key stored as `POLYGON_API_KEY` environment variable

---

## 📅 Version History

- **v1.0.0** (2026-01-18): Phase 1 implementation complete, entering stabilization period

---

**Last Updated:** 2026-01-18  
**Current Phase:** Phase 1 (Stabilization)  
**Next Milestone:** PHASE 1 CONFIRMED after 2-3 days of stable operation