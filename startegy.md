# Strategy: Hybrid AI-Assisted Change Detection Pipeline

## Implementation Update - 2026-05-07

Module 3 was moved closer to the planned dual-path Sensor architecture.

Completed in the current backend pass:

- `backend/change_types.py` is now the shared public type layer for Module 3.
  It defines `ChangeType`, `ChangeDetection`, and `ChangeDetectionResult`,
  while keeping backward-compatible aliases such as `bounding_box`,
  `detections`, `alignment`, and `diagnostics`.
- `backend/change_detection.py` now uses the shared public types and returns a
  structured result with `ground_candidates`, `object_candidates`,
  `final_detections`, `rejected_candidates`, and `debug_info`.
- Path A remains the classical ground candidate generator. It produces
  `GROUND_CHANGE` candidates from patch-based texture/structure differences,
  not final truth decisions.
- Path B now supports the object-detector adapter flow:
  detect objects on the reference image, detect objects on the aligned new
  image, compare them by IoU and class, and emit `OBJECT_ADDED` or
  `OBJECT_REMOVED`.
- In strict no-AI mode, Path B can still use the existing deterministic
  structural fallback so older deterministic tests and workflows continue to
  work without YOLO.
- `backend/object_detector.py` contains the adapter interface, `NoOpDetector`,
  and optional `YoloDetector`. The YOLO import is dynamic and guarded, so the
  no-AI scanner no longer sees a forbidden static `ultralytics` import.
- Fusion now suppresses an overlapping ground candidate only when the object
  candidate is high confidence. Low-confidence object candidates no longer
  erase ground candidates.
- `run_dual_path_change_detection(...)` now skips tracker updates on
  `GEOMETRY_FAILURE`, so bad geometry frames do not increment or break temporal
  persistence.
- Tests were expanded for object added, object removed, unchanged matched
  objects, low-confidence fusion behavior, and geometry-failure tracker
  handling.

Current verification status:

- `python test_dual_path_change_detection.py`: 18/18 passed.
- `python test_module2.py`: 6/6 passed.

## Implementation Update - Module 2 (Filter.py) Refactor

- Module 2 was refactored to use adaptive entropy-based structure weighting, replacing static weighting.
- Highlight suppression (via inpainting) and Weber contrast for LBP in shadows were added to combat difficult illumination.
- The pipeline now correctly integrates `Module2Processor` directly, extracting invariant structural maps alongside the classical candidate bounding boxes.

Still future work:

- Add a real verifier model over candidate crops.
- Enable YOLO only in an explicit hybrid-AI runtime mode with installed
  optional dependencies.
- Add richer persisted debug image export behind config.
- Add semantic reporting and optional segmentation after candidate
  verification.

## מטרת המסמך

המסמך הזה מגדיר את אסטרטגיית ההמשך למערכת Eagle-Eye. המטרה היא לשפר את איכות זיהוי השינויים, להפחית false positives, ולתת למשתמש תוצאה שימושית יותר מאשר מלבנים כלליים של `Ground Change` או `Object`.

הכיוון המומלץ הוא לא להחליף את האלגוריתם הקלאסי, אלא להפוך אותו לשכבת זיהוי ראשונית שמייצרת אזורים חשודים. לאחר מכן, שכבות חכמות יותר יאמתו, יסווגו, ויעקבו אחרי השינויים לאורך זמן.

## הבעיה במערכת הקיימת

המערכת הנוכחית היא deterministic/classical בלבד. היא מבוססת על:

- יישור גיאומטרי עם ORB ו-RANSAC.
- השוואת טקסטורה, קצוות, SSIM, entropy ו-ZNCC.
- חוקים גיאומטריים כמו compactness ו-gradient density.

הגישה הזו טובה כבסיס, אבל רגישה מאוד לבעיות בעולם אמיתי:

- שינויי תאורה.
- צללים.
- החזרי אור.
- תזוזת מצלמה קטנה.
- שגיאת alignment.
- תנועה של צמחייה.
- רעש או compression artifacts.
- הבדלים בשוליים אחרי warping.

לכן האלגוריתם הקלאסי לא צריך להיות אחראי על ההחלטה הסופית האם מדובר בשינוי אמיתי. הוא צריך להיות אחראי בעיקר על recall גבוה: לא לפספס אזורים חשודים.

## עיקרון מרכזי

המערכת צריכה לעבור מגישה של:

```text
classical detector -> final decision
```

לגישה היברידית:

```text
classical detector -> candidate regions -> AI verifier -> semantic classifier -> temporal confirmation
```

במילים אחרות:

- האלגוריתם הקלאסי מציף חשדות.
- מודל אימות מסנן false positives.
- מודל סיווג נותן משמעות סמנטית לשינוי.
- temporal tracker מאשר שהשינוי יציב לאורך זמן.

## ארכיטקטורה מומלצת

```text
Before Image
After Image
    |
    v
Module 1: Geometry Alignment
ORB + RANSAC + Homography
    |
    v
Module 2: Preprocessing
Illumination / Shadow / Structure Maps
    |
    v
Classical Candidate Generator
Ground candidates + Object candidates
    |
    v
AI Verifier
Real change vs false positive
    |
    v
Semantic Object Detector
car / person / equipment / debris / unknown
    |
    v
Temporal Tracker
3-frame persistence
    |
    v
Final Report
```

## שלב 1: להפוך את האלגוריתם הקיים ל-Candidate Generator

האלגוריתם הקיים לא צריך להחזיר “שינוי סופי”, אלא רשימת candidates.

כל candidate צריך לכלול:

- `bounding_box`
- `mask` אם קיימת
- `candidate_type`: ground/object/unknown
- `classical_score`
- `alignment_score`
- `reason`: למה האזור סומן
- `before_crop`
- `after_crop`
- `diff_crop`

מטרת השלב:

```text
למצוא אזורים חשודים, גם במחיר של false positives מסוימים.
```

האלגוריתם הקלאסי יישאר חשוב כי הוא:

- מהיר.
- דטרמיניסטי.
- לא דורש training.
- מספק geometry ו-signal features טובים.
- מצמצם את שטח החיפוש עבור מודלים כבדים יותר.

## שלב 2: AI Verifier להפחתת False Positives

זה השלב החשוב ביותר לשיפור איכות התוצאה.

ה-verifier יקבל עבור כל candidate שלושה קלטים:

```text
before crop
after crop
diff crop
```

והוא יחזיר:

```text
real_change
false_positive
```

או בצורה מפורטת יותר:

```text
real_object_added
real_object_removed
ground_change
shadow
illumination_change
alignment_artifact
vegetation_motion
noise
```

למה זה חשוב:

הקוד הקלאסי יודע לזהות הבדל. הוא לא תמיד יודע להבין האם ההבדל חשוב. מודל verifier יכול ללמוד דוגמאות של שינויים אמיתיים מול רעשים חוזרים.

מדד הצלחה מרכזי:

```text
להוריד false positives בלי לפגוע יותר מדי ב-recall.
```

## שלב 3: Semantic Object Detection

לאחר ש-candidate עבר אימות ונחשב שינוי אמיתי, צריך להבין מה השתנה.

כאן אפשר לשלב Object Detection, לדוגמה:

- YOLO
- RT-DETR
- DETR
- מודל ONNX מותאם

המטרה היא לעבור מפלט כללי:

```text
Object
```

לפלט שימושי:

```text
car
person
box
equipment
debris
unknown object
```

חשוב: Object Detector לא חייב לרוץ על כל התמונה. עדיף להריץ אותו על:

- ה-after crop.
- או candidate crop מורחב.

זה חוסך זמן ומקטין רעש.

## שלב 4: Temporal Tracking

גם אחרי אימות וסיווג, שינוי לא צריך להיות מאושר מיד בפריים אחד.

הכלל המומלץ:

```text
Real Change only if it appears in the same reference coordinates for N consecutive frames.
```

ברירת מחדל:

```text
N = 3
IoU threshold = 0.35
```

ה-tracker צריך לעבוד במערכת הקואורדינטות של תמונת הייחוס, לא בתמונה המקורית אחרי תזוזה.

תפקידו:

- לסנן רעשים רגעיים.
- למנוע אישור שגוי מתזוזה חד-פעמית.
- לוודא שהשינוי יציב.

## שלב 5: Segmentation אופציונלי

Segmentation לא צריך להיות השלב הראשון. הוא מוסיף מורכבות חישובית ואינטגרטיבית.

כדאי לשלב אותו רק אחרי שהמערכת כבר יודעת:

- למצוא candidates.
- לאמת real/false.
- לסווג סמנטית.

מודלים אפשריים:

- SAM
- FastSAM
- MobileSAM

תפקיד segmentation:

- להחזיר mask מדויק במקום bounding box גס.
- למדוד שטח שינוי.
- להפיק דוח ויזואלי מקצועי יותר.

## שלב 6: Change Detection Model ייעודי

בשלב מתקדם יותר ניתן לבדוק מודלים ייעודיים ל-change detection:

- ChangeFormer
- BIT
- DSIFN
- STANet
- Siamese U-Net

מודלים כאלה מקבלים שתי תמונות:

```text
before image
after image
```

ומחזירים:

```text
change mask
```

אבל זה שלב מתקדם, כי בדרך כלל נדרש:

- dataset מתאים.
- weights מתאימים.
- fine-tuning.
- בדיקת התאמה לסוג התמונות בפרויקט.

## שינוי הגדרת הפרויקט

אם משלבים מודלים, כבר לא נכון להציג את המערכת כ:

```text
No-AI deterministic detector
```

הגדרה מדויקת יותר:

```text
Hybrid AI-Assisted Change Detection Pipeline
```

כלומר:

מערכת היברידית לזיהוי שינויים, שמשלבת עיבוד תמונה קלאסי עם מודלי AI לצורך אימות, סיווג, והפחתת התרעות שווא.

המשמעות הטכנית:

- צריך לעדכן את `requirements.txt`.
- צריך לשנות או להסיר את `no_ai_guard.py`.
- צריך לעדכן את `DUAL_PATH_CHANGE_DETECTION.md`.
- צריך לעדכן tests.
- צריך להפריד בין מצב `classical-only` לבין מצב `hybrid-ai`.

## Roadmap מומלץ

### Phase 1: Candidate Generator

מטרה:

להפוך את `DualPathChangeDetector` לשכבה שמחזירה candidates עשירים, לא החלטה סופית בלבד.

משימות:

- להוסיף `mask` לכל candidate.
- להוסיף `reason`.
- לשמור `before_crop`, `after_crop`, `diff_crop`.
- להוסיף debug export ל-candidates.

### Phase 2: Dataset ל-Verifier

מטרה:

לבנות dataset קטן של candidates מתויגים ידנית.

מבנה מומלץ:

```text
verifier_dataset/
  real_change/
  shadow/
  illumination_change/
  alignment_artifact/
  vegetation_motion/
  noise/
```

כל דוגמה צריכה לכלול:

```text
before.png
after.png
diff.png
metadata.json
```

### Phase 3: Verifier Model

מטרה:

להוסיף מודל שמחליט האם candidate הוא שינוי אמיתי או false positive.

אפשרויות:

- מודל classification קטן.
- Siamese network.
- מודל ONNX מוכן.
- fine-tuned lightweight CNN.

פלט:

```text
verifier_label
verifier_confidence
```

### Phase 4: Semantic Detection

מטרה:

לתת משמעות לשינוי מאומת.

פלט:

```text
semantic_label
semantic_confidence
```

דוגמה:

```text
real_change=True
semantic_label=car
confidence=0.87
```

### Phase 5: Temporal Confirmation

מטרה:

לאשר שינוי רק אחרי עקביות לאורך זמן.

משימות:

- להשתמש ב-ChangeTracker הקיים.
- להוסיף שמירת track IDs.
- להוסיף final status:

```text
candidate
verified
real_persistent_change
```

### Phase 6: Segmentation

מטרה:

להחליף bounding boxes גסים במסכות מדויקות.

משימות:

- להריץ segmentation על verified candidates בלבד.
- לשמור mask.
- להציג overlay איכותי בדמו.

## מדדי הצלחה

צריך למדוד את המערכת עם מדדים ברורים:

- `precision`: כמה מהסימונים באמת נכונים.
- `recall`: כמה מהשינויים האמיתיים נמצאו.
- `false positive rate`: כמה סימונים שגויים לתמונה.
- `geometry failure rate`: כמה זוגות נכשלים ביישור.
- `average detections per image`: האם המערכת מציפה יותר מדי סימונים.
- `runtime`: זמן עיבוד לכל זוג תמונות.

יעד ראשוני טוב:

```text
להוריד false positives לפחות ב-50% ביחס לגרסה הקלאסית,
בלי להוריד recall ביותר מ-10%-15%.
```

## החלטה מומלצת להמשך

השלב הבא לא צריך להיות segmentation ולא מודל change detection כבד.

השלב הבא המומלץ הוא:

```text
AI verifier over classical candidates
```

זו הנקודה שבה נקבל את השיפור הכי משמעותי ביחס למאמץ:

- פחות false positives.
- פחות רעש ויזואלי.
- שמירה על היתרונות של האלגוריתם הקלאסי.
- בסיס טוב להמשך semantic detection.

## סיכום

האלגוריתם הקלאסי הוא בסיס טוב, אבל הוא לא צריך להיות המחליט הסופי. הדרך הנכונה להמשך היא מערכת היברידית:

```text
Classical CV for candidate generation
AI verifier for false-positive reduction
Object detector for semantic meaning
Temporal tracker for final confirmation
Optional segmentation for precise masks
```

כך המערכת תעבור ממנגנון פשוט של image difference למערכת יציבה ושימושית יותר לזיהוי שינויים בעולם אמיתי.
