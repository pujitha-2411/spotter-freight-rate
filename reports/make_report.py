# Builds reports/freight_rate_report.pdf from metrics.json and the figures.
# Run from the repo root after src.train and score.py:  python reports/make_report.py
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from PIL import Image as PI
M=json.load(open('reports/metrics.json'))
ss=getSampleStyleSheet(); H1=ss['Title']; H2=ParagraphStyle('h2',parent=ss['Heading2'],textColor=colors.HexColor('#064A56'))
B=ParagraphStyle('b',parent=ss['BodyText'],fontSize=10,leading=14); S=ParagraphStyle('s',parent=B,fontSize=8.5,textColor=colors.grey)
def img(p,w=16):
    W,Hh=PI.open(p).size; return Image(p,width=w*cm,height=w*cm*Hh/W)
def tbl(rows,widths=None):
    t=Table(rows,colWidths=widths); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#064A56')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
      ('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#C9D4D6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F6')])])); return t
b=M['baseline_distance_only']; g=M['trend_plus_lightgbm']
el=[Paragraph('Freight Rate Prediction: Validation &amp; Split Approach',H1),
Paragraph('Pujitha Kolakonda Pusala &middot; Spotter AI, Machine Learning Engineer assessment &middot; October 2026',S),Spacer(1,8),
Paragraph('Summary',H2),
Paragraph(f"The final model predicts <b>log(rate per mile)</b> with a small linear time trend plus LightGBM. On a time-based holdout (train Jan&ndash;Aug 2025, test Sep&ndash;Oct 2025) it scores <b>MAPE {g['MAPE_all']}%</b> on all loads and <b>{g['MAPE_believable']}%</b> on loads with a believable rate, against {b['MAPE_all']}% / {b['MAPE_believable']}% for a distance-only baseline. Almost all remaining error comes from ~1.4% of loads whose recorded rate looks like a data-entry error.",B),
Paragraph('1. Data and what the EDA showed',H2),
Paragraph("train_test.csv has 48,000 labelled loads from 2025-01-01 to 2025-10-31; validation.csv has 12,000 unlabelled loads from 2025-11-01 to 2025-12-31. Key findings: (a) price is almost linear in distance (correlation 0.91), median about $2.15 per mile; (b) Reefer costs more per mile than Flatbed, then Dry Van; (c) prices follow market_index (a daily value) up to a June peak and back down; (d) 8 validation cities (12.1% of validation loads) never appear in training.",B),
Paragraph('2. Data-quality issues and fixes',H2),
tbl([['Issue','Train','Validation','Fix'],
 ['Negative weight (-5,000 to -47,500 lb)','292','145','Flip sign: same shape as positive weights'],
 ['Missing weight','300','165','Median weight for that equipment (from training)'],
 ['Missing market_index','374','249','Median of the same day (it varies ~0.02 within a day)'],
 ['Rate/mile < $1.30 or > $4.00','677 (1.4%)','n/a','Dropped from training only; still scored in validation']],[6*cm,2*cm,2.2*cm,6.3*cm]),Spacer(1,4),
Paragraph("The two outlier groups are separated from the healthy range ($1.67&ndash;$3.18 per mile, 1st&ndash;99th percentile) by empty gaps and show no pattern by equipment, weekday, month or city, so they look like random entry errors rather than real prices. The same feature fixes run on validation, because it has the same problems.",B),
PageBreak(),
Paragraph('3. How the data was split and validated',H2),
Paragraph("<b>Why time-based.</b> The loads we must predict (Nov&ndash;Dec) are entirely after the training period. A random split would let the model see neighbouring days from the test period and look better than it really is. So every check trains on the past and tests on the future.",B),
Paragraph("<b>Main holdout.</b> Train on Jan&ndash;Aug 2025 (the model and the trend are fitted only on these rows), test on Sep&ndash;Oct 2025. Sep&ndash;Oct is the closest period to Nov&ndash;Dec and has a similar market level (market_index ~0.93 vs 0.92&ndash;0.94 in validation).",B),
Paragraph("<b>Rolling backtest.</b> To check that one lucky split isn't driving the result, the model is refitted for each of Jul, Aug, Sep and Oct using only the months before it.",B),
Paragraph("<b>Metrics.</b> MAE in dollars and MAPE in percent, on all test rows and on rows with a believable rate. Outliers are removed from training but never from the scored test rows.",B),Spacer(1,4),
tbl([['Model / test period','MAE all','MAPE all','MAE believable','MAPE believable']]+
 [['Baseline (distance only), Sep-Oct',f"${b['MAE_all']}",f"{b['MAPE_all']}%",f"${b['MAE_believable']}",f"{b['MAPE_believable']}%"],
  ['Trend + LightGBM, Sep-Oct',f"${g['MAE_all']}",f"{g['MAPE_all']}%",f"${g['MAE_believable']}",f"{g['MAPE_believable']}%"]]+
 [[f"Backtest {r['test_month']}",f"${r['MAE_all']}",f"{r['MAPE_all']}%",f"${r['MAE_believable']}",f"{r['MAPE_believable']}%"] for r in M['rolling_backtest']],
 [5.6*cm,2.2*cm,2.2*cm,2.8*cm,3*cm]),Spacer(1,4),
Paragraph("August is the hardest month: market_index fell sharply (1.20 in July to 0.98) but prices fell much less, a regime the earlier months hadn't shown. The model adapts once August is in training, which is why Sep and Oct are back to 1.2&ndash;1.9%. After validation, the final model is refitted on all of Jan&ndash;Oct before predicting Nov&ndash;Dec.",B),
KeepTogether([img('reports/figures/05_holdout_pred_vs_actual.png',9.5),Paragraph('Holdout Sep&ndash;Oct: believable loads sit on the diagonal; the red points are the recorded-rate errors.',S)]),
PageBreak(),
Paragraph('4. Model',H2),
Paragraph("<b>Target:</b> log(posted_rate / distance). Dividing by distance removes the biggest effect; the log makes errors relative and guarantees positive predictions. <b>Features:</b> road and straight-line miles, circuity, pickup and delivery coordinates (instead of city names, so unseen cities still work), equipment, weight, market_index, quote_signal, the day's average quote signal and the load's quote relative to it, and day of week. Month and day-of-year are left out because Nov&ndash;Dec never appear in training.",B),
Paragraph(f"<b>Trend + trees:</b> after accounting for lane, equipment and market, rates creep up about <b>{M['trend_percent_per_month']}% per month</b>. Tree models can't extrapolate beyond the last date they saw, so a linear regression estimates this one slope and LightGBM is trained on the de-trended target. Adding the trend cut Sep&ndash;Oct believable-row MAPE from about 2.1% to 1.5% in experiments. Comparing quote_signal with the same day's average was the other big gain (about 3.4% to 3.0% before the trend).",B),
img('reports/figures/06_feature_importance.png',11.5),
Paragraph('5. December chart (from score.py)',H2),
Paragraph("The December file only gives lane, distance, equipment, weight and date. Coordinates come from the city names in training; market_index and quote signal are the averages of the real validation loads on each December day. The line therefore follows the weekly market cycle seen in the data, with a slight upward drift, around $830&ndash;$850 for this 360-mile Dry Van lane (historically ~$2.32/mile on Lexington&rarr;Fort Wayne).",B),
img('scorer_results/candidate_december.png',16.5),
Paragraph('6. Limitations and next steps',H2),
Paragraph("Training has no November or December, so holiday effects (Thanksgiving, Christmas capacity crunch) can only enter through market_index; if real holiday premiums exist, the December line underestimates them. With more history I would add year-over-year seasonality, quantile models for a price range rather than one number, and an outlier classifier if the data errors turn out to follow a pattern.",B)]
SimpleDocTemplate('reports/freight_rate_report.pdf',pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=1.6*cm,bottomMargin=1.6*cm,title='Freight Rate Prediction Report').build(el)
