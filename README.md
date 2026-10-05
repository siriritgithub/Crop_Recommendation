# 🌾 Smart Agri Assistant

A data-science portfolio project for smallholder farmers in India: crop
recommendation, crop price prediction, agronomy guidance, a profit
calculator, and a retrieval-grounded farming chatbot — all in one
Streamlit app.

**Live demo:** _add your Streamlit Community Cloud link here after deploying_

## Why this project

Built to practice the full lifecycle of a data science / applied-AI
project: cleaning real data, training and honestly evaluating ML models,
recognizing when a model *isn't* trustworthy (see "Design decisions"
below), and shipping an interactive app — plus a small retrieval-augmented
LLM feature to apply what I've been learning about GenAI application
building.

## What's in the app

| Tab | What it does | Powered by |
|---|---|---|
| 🌱 Crop Recommendation | Top-3 crop matches for given soil/climate inputs, with a plain-language reason and suitable growing season for each | Random Forest classifier, trained on 2,200 real records, 22 crops |
| 💰 Price Prediction | Predicts a commodity's national-average price for a given month/year, in your choice of quintal/kg/ton, with an increase/decrease trend vs. the latest known price | Random Forest regressor, trained on real historical mandi price data (2,800+ monthly records, 16 commodities) |
| 📊 Profit Calculator | Cost vs. revenue vs. profit for a given crop, area, and yield | Simple business logic, pre-fillable from the price prediction |
| 🧪 Fertilizer & Pest Guidance | NPK dosage (switchable kg/ha or kg/acre), application timing, and before/after safety precautions per crop | Curated agronomy reference (5 major crops) + a data-derived estimate for the rest |
| 💧 Irrigation Advisor | Rule-based watering guidance from rainfall, temperature and soil type | Rule-based logic + optional live temperature |
| 🏛 Government Schemes | Schemes by crop and state, covering all 28 states + 8 union territories (falls back to national schemes where no state-specific entry is curated yet), plus real private-sector programs | Curated reference data |
| 🤖 AI Farm Assistant | Chat-based Q&A; grounds scheme/fertilizer answers in this app's own data before answering. Falls back to a small built-in keyword responder if no API key is set, so the tab always works | Anthropic API + lightweight keyword-retrieval step |
| 📝 Feedback | Star rating + comments, logged locally with a live ratings chart | Local CSV log |

Also supports English / Telugu / Tamil UI text.

## Model performance (honest numbers, not vibes)

Run the training scripts yourself to regenerate these — they're also
written to `models/*_metrics.json` after each run.

- **Crop recommendation:** ~99% test accuracy, 5-fold cross-validated,
  22 crop classes. (This dataset's classes are cleanly separable by
  N/P/K/climate — a real-world deployment on messier field data would
  likely see lower accuracy; that caveat is worth saying out loud in an
  interview, not hiding.)
- **Price prediction:** R² ≈ 0.87, MAE ≈ ₹611/quintal on held-out
  historical data. Predicts *national average* price trends
  (seasonality + year-over-year trend) — it is not a real-time mandi
  price feed and shouldn't be read as one.

## Design decisions worth knowing about

This project started from an earlier version with some issues I found
and fixed while reviewing it — noting them here because *why* something
was changed is more interesting in an interview than pretending it was
always this way:

- **The old price model was fit to fake data.** It took two UI sliders
  ("market demand", "supply") that the user just drags, with no real
  data behind them, and learned to predict price from that. It ran
  without erroring, but it wasn't predicting from anything real. It's
  now trained on the actual historical price dataset instead.
- **Soil type is not a model input.** The source crop dataset's
  `soil_type` column is ~99.8% missing (2,195 of 2,200 rows). Rather
  than silently imputing a nearly-empty column and letting it *look*
  like it's informing predictions, it's excluded from the model
  entirely. It's still used as a genuine rule-based input in the
  Irrigation tab, where clay/sandy/loamy is meaningful without needing
  to be "predicted."
- **Fertilizer suggestion is rule-based, not a black-box classifier.**
  The old fertilizer model had no training script anywhere in the
  project and no verifiable data behind its N/P/K → fertilizer-type
  mapping. An unexplainable model is worse than a transparent rule
  table, so this app uses curated agronomy reference values (with a
  clearly-labeled data-derived fallback for crops outside the curated
  set) instead.
- **Live weather is used for temperature/humidity only, never
  "rainfall."** The model's rainfall feature is a *seasonal cumulative
  total* (20–300mm in the training data); a live weather API only
  reports rain in the last hour (basically always ~0). Feeding that in
  would have silently pushed every single prediction to the extreme
  edge of the training distribution. Rainfall stays a manual input.

## Second round of fixes (from real usage notes)

After the first pass, further hands-on testing surfaced a second set of
issues — the kind that only show up once you're actually poking at the
running app, which is exactly the right way to find them:

- **Weather lookups weren't cached.** Every slider tweak reruns the whole
  Streamlit script, so without caching, the same city got re-fetched from
  the API on every single interaction — the actual cause of the app
  feeling slow, and a source of what looked like a "stale value" bug.
  Fixed with a 10-minute cache keyed on the exact city string.
- **Small towns/villages failing weather lookup.** Appending the `,IN`
  country code to the query measurably improves match rate; a manual
  slider fallback was already there and still is.
- **Dropdowns were too rigid.** Soil type, scheme state, and scheme crop
  all now have an "Other (not listed)" option that reveals a free-text
  field, rather than forcing a wrong pick.
- **"What does target year/month mean?" in Price Prediction** — added
  inline help text, plus a per-quintal/kg/ton unit toggle and an
  increase/decrease trend readout versus the latest known price.
  State-level pricing was requested too, but the source dataset has
  *zero* state-level granularity (verified directly — every row is
  `state='India'`), so rather than fabricate state numbers, the app says
  plainly that this is a national average.
- **"What does kg/ha mean, I'm a small farmer"** — Fertilizer tab now has
  an acre/hectare toggle, plus explicit application timing and
  before/after safety precautions (for both the crop and the person
  applying it) for every curated crop.
- **Government Schemes only covered 6 states.** The state dropdown now
  lists all 28 states + 8 union territories; ones without a curated
  state-specific entry fall back to the national schemes instead of
  showing nothing. Added a few real private-sector programs (ITC
  e-Choupal, IFFCO Kisan, DeHaat, Samunnati) with a filter toggle, and a
  "my crop isn't listed" path that still surfaces the national schemes.
  A **live government scheme API was asked for but doesn't exist** in a
  form this project can call — schemes stay curated reference data,
  labeled as such.
- **Chatbot tab required an API key to do anything.** It now falls back
  to a small built-in keyword responder when no key is set, so the tab
  always works, with the full LLM+retrieval experience unlocked by adding
  a key.
- **"Streamlit feels slow, should we switch frameworks?"** The actual
  cause (uncached weather calls re-firing on every rerun) is fixed above.
  A full framework swap to something like FastAPI + React would be real
  engineering effort with no data-science payoff for a portfolio piece,
  so that wasn't done — same reasoning as the earlier decision to not
  rebuild the stack.

## Cut from an earlier, more ambitious scope

An earlier plan for this project included user accounts, offline mode,
voice input/output, and a CNN-based pest/disease image detector. Those
are deliberately left out here:

- **Auth + saved history** — real backend/security work with no data
  science payoff; not what this project is meant to demonstrate.
- **Offline mode** — infrastructure-heavy, not a differentiator for a
  DS/AI portfolio piece.
- **Voice input/output** — a nice demo feature, but a poor use of time
  relative to the ML/GenAI work this project is meant to showcase.
- **Pest/disease detection via image upload (CNN)** — needs its own
  image dataset and training pipeline; it deserves to be its own
  computer-vision project rather than a bolted-on tab here.

## Project structure

```
agri-assistant/
├── app.py                     # single Streamlit entry point
├── data/
│   ├── Crop_recommendation.csv     # 2,200 rows, 22 crops
│   └── crop_price_dataset.csv      # historical monthly mandi prices
├── models/                    # generated by the training scripts below
├── src/
│   ├── data_preprocessing.py
│   ├── train_crop_model.py
│   ├── train_price_model.py
│   ├── fertilizer_recommender.py
│   ├── govt_schemes.py
│   ├── translations.py
│   └── chatbot.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── requirements.txt
```

## Running it locally

```bash
pip install -r requirements.txt

# Train the models (writes to models/)
python src/train_crop_model.py
python src/train_price_model.py

# Copy the secrets template and (optionally) add your own API keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

The app runs fully without any API keys — live weather lookup and the
chatbot tab just show a short message asking for a key instead of
erroring.

## Future work

- Retrain the crop model with real (not near-empty) soil-type data if a
  better source dataset becomes available.
- Curated fertilizer/pest guidance for more of the 22 crops.
- A real vector-store RAG pipeline (currently keyword-based) if the
  knowledge base grows beyond what keyword retrieval can handle well.
- Pest/disease detection as its own separate computer-vision project.

## Author

Sirisha — Data Science student focused on Data Analytics, ML, and
Generative AI. See other projects: Retail Sales Analytics (MySQL +
Power BI) and FinWise (personal finance app).
