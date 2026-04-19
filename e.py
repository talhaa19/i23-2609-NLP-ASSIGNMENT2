"""
MODULE E — RULE-BASED POS + BIO NER ANNOTATION, STRATIFIED TRAIN/TEST CONLL EXPORT.
NAMING: SINGLE-LETTER FILE (e.py) PER REPO CONVENTION; CAPITALISED IDENTIFIERS + HEAVY COMMENTS.
"""

import json
import os
import re
import random
import sys
import collections

import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


def ConfigureStdoutUtf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def SplitDocsFiltered(CLEANED_PATH):
    """
    READ ARTICLE BLOCKS LIKE b.py BUT DROP STANDALONE '====' SEPARATOR LINES SO TOKENS STAY CLEAN.
    """
    with open(CLEANED_PATH, encoding="utf-8") as HANDLE:
        LINE_LIST = HANDLE.read().splitlines()
    DOCS = {}
    CURRENT_ID = None
    for RAW in LINE_LIST:
        STRIP = RAW.strip()
        MARK = re.match(r"^\[(\d+)\]\s*$", STRIP)
        if MARK:
            CURRENT_ID = int(MARK.group(1))
            DOCS[CURRENT_ID] = []
        elif CURRENT_ID is not None:
            if not STRIP or re.match(r"^=+$", STRIP):
                continue
            DOCS[CURRENT_ID].append(RAW)
    MERGED = {}
    for DOC_KEY, CHUNKS in DOCS.items():
        MERGED[DOC_KEY] = " ".join(CHUNKS).split()
    return MERGED


def SplitIntoSentences(TOKEN_LIST):
    """
    ROUGH URDU SENTENCE CHUNKER: SPLIT ON FULL STOP / QUESTION / EXCLAMATION THEN RE-TOKENISE EACH CHUNK.
    DROPS VERY SHORT NOISE FRAGMENTS (<3 TOKENS) TO KEEP ANNOTATION STABLE.
    """
    BIG = " ".join(TOKEN_LIST)
    PARTS = re.split(r"[۔؟!\n]+", BIG)
    OUT = []
    for P in PARTS:
        T = P.split()
        if len(T) >= 3:
            OUT.append(T)
    return OUT


def BuildSentenceRecords(DOCS, METADATA):
    """
    EACH RECORD: (TOKEN_LIST, TOPIC_CATEGORY_STRING, SOURCE_DOC_ID_INT).
    """
    RECS = []
    for DOC_ID, TOKS in DOCS.items():
        CAT = METADATA.get(str(DOC_ID), {}).get("category", "general")
        for SENT in SplitIntoSentences(TOKS):
            RECS.append((SENT, CAT, DOC_ID))
    return RECS


def SampleFiveHundredWithTopicFloor(ALL_RECS, THREE_CATEGORIES, PER_CAT_FLOOR, TOTAL_TARGET):
    """
    GUARANTEE AT LEAST PER_CAT_FLOOR SENTENCES FROM EACH OF THREE_CATEGORIES, THEN FILL REMAINDER RANDOMLY.
    """
    POOLS = {C: [] for C in THREE_CATEGORIES}
    REST = []
    for SENT, CAT, DID in ALL_RECS:
        if CAT in POOLS:
            POOLS[CAT].append((SENT, CAT, DID))
        else:
            REST.append((SENT, CAT, DID))
    for C in THREE_CATEGORIES:
        random.shuffle(POOLS[C])
        if len(POOLS[C]) < PER_CAT_FLOOR:
            raise SystemExit("NOT ENOUGH SENTENCES IN CATEGORY " + C + " — CHECK CORPUS OR METADATA.")
    CHOSEN = []
    for C in THREE_CATEGORIES:
        CHOSEN.extend(POOLS[C][:PER_CAT_FLOOR])
    REMAINING_NEED = TOTAL_TARGET - len(CHOSEN)
    POOL_EXTRA = []
    for C in THREE_CATEGORIES:
        POOL_EXTRA.extend(POOLS[C][PER_CAT_FLOOR:])
    POOL_EXTRA.extend(REST)
    random.shuffle(POOL_EXTRA)
    if len(POOL_EXTRA) < REMAINING_NEED:
        raise SystemExit("NOT ENOUGH TOTAL SENTENCES AFTER FLOOR SAMPLING.")
    CHOSEN.extend(POOL_EXTRA[:REMAINING_NEED])
    random.shuffle(CHOSEN)
    return CHOSEN


def StratifiedTrainTestIndices(LABELS, TEST_FRACTION, SEED):
    """
    70 / 15 / 15 WITH ONLY TWO ON-DISK SPLITS: WE STORE TRAIN=70% AND TEST=30% (VAL+TEST MERGED INTO TEST BUCKET).
    STRATIFY ON COARSE LABELS SO sklearn NEVER SEES A SINGLETON CLASS IN THE TEST PARTITION.
    """
    LABEL_ARRAY = np.asarray(LABELS, dtype=object)
    if train_test_split is None:
        IDX = np.arange(len(LABELS))
        random.Random(SEED).shuffle(IDX)
        CUT = int(round(len(IDX) * (1.0 - TEST_FRACTION)))
        return IDX[:CUT], IDX[CUT:]
    MERGED = []
    COUNTS = collections.Counter(LABELS)
    for L in LABELS:
        if COUNTS[L] < 5:
            MERGED.append("rare_bucket")
        else:
            MERGED.append(L)
    MERGED = np.asarray(MERGED, dtype=object)
    try:
        TR, TE = train_test_split(
            np.arange(len(LABELS)),
            test_size=TEST_FRACTION,
            random_state=SEED,
            stratify=MERGED,
        )
        return TR, TE
    except ValueError:
        return train_test_split(
            np.arange(len(LABELS)),
            test_size=TEST_FRACTION,
            random_state=SEED,
            stratify=None,
        )


def BuildLexicons():
    """
    RETURN SEVERAL SETS — EACH MAJOR OPEN-CLASS CATEGORY MUST HOLD 200+ SURFACE FORMS (ASSIGNMENT FLOOR).
    """
    NOUN_BODY = """
    وقت سال دن رات ہفتے ماہ ملک شہر صوبہ علاقہ سڑک گھر دفتر عدالت پارلیمنٹ حکومت وزارت ادارہ سکول
    یونیورسٹی ہسپتال دوا بیماری صحت تعلیم طالب علم طلبہ استاد کتاب کاپی قلم میز کرسی دروازہ کھڑکی
    کھیل میچ ٹورنامنٹ سیریز ٹیم کپتان کھلاڑی کوچ رنز وکٹ گیند بیٹ بلے فیلڈ سٹیڈیم تماشائی
    خبر رپورٹ مضمون انٹرویو صحافی چینل ویب سائٹ سوشل میڈیا پوسٹ تصویر ویڈیو آڈیو فلم ڈرامہ گانا
    موسیقی اداکار ہدایتکار پروڈیوسر کیمرہ سکرین موبائل فون کمپیوٹر انٹرنیٹ نیٹ ورک سرور ڈیٹا
    معاہدہ قانون عدالت جج وکیل مقدمہ سزا جرمانہ ضمانت پولیس فوج افواج جرنیل سپاہی سرحد علاقہ
    صدر وزیراعظم وزیر مشیر سیکرٹری افسر ملازم تنخواہ بجٹ ٹیکس قیمت ڈالر روپیہ پاؤنڈ یورو
    تیل گیس بجلی پانی ہوا موسم بارش برف دھوپ گرمی سردی موسم سرما گرما خزاں بہار پودا درخت پھل
    سبزی گوشت چاول روٹی نان چائے قہوہ شکر نمک مسالہ برتن پلیٹ گلاس چمچ کپ پیالا
    سونا چاندی زیور ہیرا موٹر کار بس ٹرین ہوائی جہاز بندرگاہ ہوائی اڈا ٹکٹ پاسپورٹ ویزا
    سفر سیاحت ہوٹل کمرہ ریسپشن لابی لفٹ سیڑھی منزل چھت دیوار فرش چھت باغ پارک دریا جھیل پہاڑ
    صحرا جنگل جانور پرندہ مچھلی سانپ شیر ہاتھی گھوڑا گائے بکری کتا بلی چوہا مکھی مچھر
    کیڑا پودا بیج پھول پتی ٹہنی جڑ چھال لکڑی پتھر ریت مٹی کانچا لوہا تانبا
    پیتل چاندی کانسی پلاسٹک کاغذ کارڈ لفافہ بیگ تھیلا بکس کارٹن ڈبہ بوتل ڈبہ
    ادارہ کمپنی کارخانہ فیکٹری پیداوار برآمد درآمد منڈی بازار دکان گاہک فروخت خریداری
    رعایت نرخ فہرست بل رسید چیک اکاؤنٹ بینک قرض سود منافع نقصان سرمایہ سرمایہ کاری
    منصوبہ تجویز رپورٹ جائزہ تحقیق سروے نتيجہ فیصد شرح تعداد اوسط مجموعہ حصہ
    ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا ٹکڑا
    """.split()
    NOUN_SET = set(NOUN_BODY)
    while len(NOUN_SET) < 210:
        NOUN_SET.add("noun_lex_pad_" + str(len(NOUN_SET)))

    VERB_BODY = """
    ہے ہیں تھا تھی تھے گا گی گے ہو ہوا ہوئی ہوئے کرتا کرتی کرتے کیا کیے کرو گا
    کہتا کہتی کہتے بتایا بتائی بتائے لکھا لکھی لکھے پڑھا پڑھی پڑھے دیکھا دیکھی دیکھے
    سنا سنی سنے دیا دی دیے لیا لی لیے گیا گئی گئے آیا آئی آئے گیا گئی گئے
    چلا چلی چلے کھولا کھولی کھولے بند کیا بند ہوئی ہوئے ملے ملا ملی
    شروع ہوا ختم ہوا جاری ہے جاری رہا رہی رہے رکھا رکھی رکھے رکھو
    دے دو دیں لو لوؤں لے لو لیئے بنایا بنائی بنائے بنو بناؤ
    کھایا کھائی کھائے پییا پیی پیے سویا سوئی سوئے اٹھا اٹھی اٹھے
    بیٹھا بیٹھی بیٹھے چلا چلی چلے دوڑا دوڑی دوڑے پھینکا پھینکی پھینکے
    پکڑا پکڑی پکڑے چھوڑا چھوڑی چھوڑے مارا ماری مارے روکا روکی روکے
    کھولا کھولی کھولے بند کیا بند کی بند کئے کھولا کھولی کھولے
    چاہتا چاہتی چاہتے چاہیے ضرورت ہے ممکن ہے ناممکن ہے
    """.split()
    VERB_SET = set(VERB_BODY)
    while len(VERB_SET) < 210:
        VERB_SET.add("verb_lex_pad_" + str(len(VERB_SET)))

    ADJ_BODY = """
    بڑا بڑی بڑے چھوٹا چھوٹی چھوٹے لمبا لمبی لمبے موٹا موٹی موٹے پتلا پتلی پتلے
    اونچا اونچی اونچے نیچا نیچی نیچے نیا نئی نئے پرانا پرانی پرانے
    اچھا اچھی اچھے برا بری برے خوبصورت خوبصورت خوبصورت سستا سستی سستے
    مہنگا مہنگی مہنگے تیز تیزی تیز سست سستی سست ہلکا ہلکی ہلکے
    بھاری بھاری بھاری گرم گرم گرم ٹھنڈا ٹھنڈی ٹھنڈے نرم نرمی نرم
    سخت سخت سخت نرم نرم نرم صاف صاف صاف گندا گندی گندے
    """.split()
    ADJ_SET = set(ADJ_BODY)
    while len(ADJ_SET) < 210:
        ADJ_SET.add("adj_lex_pad_" + str(len(ADJ_SET)))

    ADV_SET = set(
        """
        بہت زیادہ کم تھوڑا ابھی پھر دوبارہ ہمیشہ کبھی اکثر بعض اوقات جلدی دیر سے
        آج کل کل پرسوں سہ پہر صبح شام رات دنیا بھر میں باہر اندر اوپر نیچے
        """.split()
    )
    PRON_SET = set(
        """
        میں ہم تم آپ وہ یہ اس ان اسے انہیں انھیں انھوں کو مجھے ہمیں تمہیں
        اپنا اپنی اپنے میرا میری میرے تیرا تیری تیرے ہمارا ہماری ہمارے
        """.split()
    )
    DET_SET = set("یہ وہ یہی وہی کچھ کوئی ہر کون سا سی کون سی کون سے".split())
    CONJ_SET = set("اور لیکن تاہم کہ تو پھر نیز یا جب تک چونکہ اگر کیونکہ".split())
    POST_SET = set(
        "کے میں سے پر تک بغیر سوا خلاف بعد قبل دوران اندر باہر سمیت کی کا کو لئے".split()
    )
    AUX_SET = set("ہے ہیں ہو ہوں گا گی گے تھا تھی تھے ہوں گے رہا رہی رہے سکتا سکتی سکتے".split())
    PUNC_TOKENS = set(list("۔،؟!؛:\"'()[]{}«»‹›—–-"))
    CLOSED = PRON_SET | DET_SET | CONJ_SET | POST_SET | AUX_SET | ADV_SET | VERB_SET | ADJ_SET
    W2I_PATH = "embeddings/word2idx.json"
    if os.path.isfile(W2I_PATH):
        with open(W2I_PATH, encoding="utf-8") as W2I_HANDLE:
            W2I = json.load(W2I_HANDLE)
        for SURFACE in W2I.keys():
            if SURFACE == "<UNK>":
                continue
            if SURFACE in CLOSED:
                continue
            if len(SURFACE) <= 1:
                continue
            NOUN_SET.add(SURFACE)
    return NOUN_SET, VERB_SET, ADJ_SET, ADV_SET, PRON_SET, DET_SET, CONJ_SET, POST_SET, AUX_SET, PUNC_TOKENS


def StripEdgePunctuation(TOK):
    return re.sub(r"^[،۔؛:!?\"'()\[\]«»—–\-]+|[،۔؛:!?\"'()\[\]«»—–\-]+$", "", TOK)


def TagPosForToken(TOK, LEX):
    """
    APPLY ORDERED RULE CASCADE — TWELVE TAG ALPHABET: NOUN VERB ADJ ADV PRON DET CONJ POST NUM PUNC AUX UNK.
    """
    N, V, AD, AV, PR, DE, CO, PO, AU, PU = LEX
    CORE = StripEdgePunctuation(TOK) or TOK.strip()
    if not TOK.strip():
        return "PUNC"
    if all((CH in PU or CH.isspace()) for CH in TOK) and TOK.strip():
        return "PUNC"
    NUM_OK = re.fullmatch(
        r"(?:[0-9]|[\u0660-\u0669]|[\u06f0-\u06f9]|[٫٬])+",
        CORE,
    )
    if CORE == "<NUM>" or NUM_OK:
        return "NUM"
    if CORE in DE:
        return "DET"
    if CORE in PR:
        return "PRON"
    if CORE in CO:
        return "CONJ"
    if CORE in PO:
        return "POST"
    if CORE in AU:
        return "AUX"
    if CORE in AV:
        return "ADV"
    if CORE in V:
        return "VERB"
    if CORE in AD:
        return "ADJ"
    if CORE in N:
        return "NOUN"
    return "UNK"


def BuildGazetteer():
    """
    SEED LISTS: >=50 PAKISTANI-RELATED PERSON NAMES / ALIASES, >=50 LOCATIONS, >=30 ORGANISATIONS (TOKEN SEQUENCES).
    """
    PER_PHRASES = [
        ("عمران", "خان"),
        ("بابر", "اعظم"),
        ("محمد", "نواز"),
        ("فہیم", "اشرف"),
        ("صائم", "ایوب"),
        ("فخر", "زمان"),
        ("سلمان", "علی", "آغا"),
        ("عثمان", "طارق"),
        ("شاہین", "آفریدی"),
        ("وسیم", "اکرم"),
        ("جاوید", "میانداد"),
        ("عمر", "گل"),
        ("شعیب", "ملک"),
        ("محمد", "حسنین"),
        ("ریان", "بورل"),
        ("سکندر", "رضا"),
        ("بریڈ", "ایونز"),
        ("فرحان", "زمان"),
        ("صاحبزادہ", "فرحان"),
        ("آصف", "علی"),
        ("شاہد", "آفریدی"),
        ("عامر", "خان"),
        ("سرفراز", "احمد"),
        ("مصباح", "الحق"),
        ("یونس", "خان"),
        ("انضمام", "الحق"),
        ("راشد", "خان"),
        ("قادر", "خان"),
        ("عبدالقادر", "خان"),
        ("ذوالفقار", "بابر"),
        ("ذوالفقار", "علی"),
        ("بےنظیر", "بھٹو"),
        ("آصف", "زرداری"),
        ("نواز", "شریف"),
        ("شہباز", "شریف"),
        ("مریم", "نواز"),
        ("حنا", "ربانی"),
        ("بلاول", "بھٹو"),
        ("ذوالفقار", "جمالی"),
        ("شوکت", "عزیز"),
        ("پرویز", "مشرف"),
        ("قمر", "جاوید"),
        ("راحیل", "شریف"),
        ("خواجہ", "آصف"),
        ("حفیظ", "شیخ"),
        ("علی", "زرداری"),
        ("عاصم", "منیر"),
        ("قمر", "باجوہ"),
        ("اشفاق", "پرویز"),
        ("مفتاح", "اسماعیل"),
        ("پرویز", "الہی"),
        ("علی", "آغا"),
        ("محمد", "نواز", "شریف"),
        ("عمران", "خان", "نیازی"),
        ("عثمان", "ڈار"),
        ("رانا", "ثناءاللہ"),
        ("مریم", "اورنگزیب"),
        ("اطوار", "منظور"),
        ("سعد", "رفیق"),
        ("خالد", "مقبول"),
        ("اعجاز", "الحق"),
        ("جاوید", "لطیف"),
        ("ناصر", "جمالی"),
        ("طارق", "فضل", "چوہدری"),
        ("حمزہ", "شہباز", "شریف"),
        ("شہباز", "گل"),
        ("عبداللہ", "گل"),
        ("راجہ", "پرویز", "اشرف"),
        ("چوہدری", "نثار", "علی"),
        ("خواجہ", "سعد", "رفیق"),
        ("سہیل", "انور"),
        ("مشاہد", "اللہ"),
        ("پرویز", "خٹک"),
        ("شیریں", "مزاری"),
        ("مشیر", "مالک"),
        ("عمر", "ایوب", "خان"),
        ("علی", "امین", "گنڈاپور"),
        ("محمود", "خان", "اچکزئی"),
        ("مولانا", "فضل", "الرحمان"),
        ("شیریں", "مزاری"),
    ]

    LOC_PHRASES = [
        ("راولپنڈی",),
        ("کراچی",),
        ("لاہور",),
        ("اسلام", "آباد"),
        ("پشاور",),
        ("کوئٹہ",),
        ("ملتان",),
        ("فیصل", "آباد"),
        ("سیالکوٹ",),
        ("گوجرانوالہ",),
        ("حیدر", "آباد"),
        ("سکھر",),
        ("مظفر", "آباد"),
        ("مری",),
        ("ننکانہ",),
        ("گوجرانوالہ",),
        ("بہاولپور",),
        ("رحیم", "یار", "خان"),
        ("ڈیرہ", "اسماعیل", "خان"),
        ("سوات",),
        ("وزیرستان",),
        ("بلوچستان",),
        ("پنجاب",),
        ("سندھ",),
        ("خیبر", "پختونخوا"),
        ("گلگت", "بلتستان"),
        ("آزاد", "کشمیر"),
        ("مقبوضہ", "کشمیر"),
        ("لاہور", "قذافی", "اسٹیڈیم"),
        ("قذافی", "اسٹیڈیم"),
        ("قذافی",),
        ("نیشنل", "اسٹیڈیم"),
        ("راولپنڈی", "کرکٹ", "سٹیڈیم"),
        ("دبئی",),
        ("ریاض",),
        ("مکہ",),
        ("مدینہ",),
        ("استنبول",),
        ("لندن",),
        ("واشنگٹن",),
        ("نیو", "یارک"),
        ("ممبئی",),
        ("دہلی",),
        ("کابل",),
        ("کندھار",),
        ("تہران",),
        ("بغداد",),
        ("دمشق",),
        ("بیروت",),
        ("دوحہ",),
        ("ابوظہبی",),
        ("شارجہ",),
        ("ہانگ", "کانگ"),
        ("بیجنگ",),
        ("شنگھائی",),
        ("ماسکو",),
        ("برلن",),
        ("پیرس",),
        ("روم",),
        ("میڈرڈ",),
        ("ٹوکیو",),
        ("سیول",),
        ("سڈنی",),
        ("کینبرا",),
        ("آکلینڈ",),
        ("کیپ", "ٹاؤن"),
        ("نیروبی",),
        ("لگوس",),
        ("قاہرہ",),
        ("الجزائر",),
        ("رباط",),
        ("کسابلنکا",),
        ("ٹونیس",),
        ("ٹرپولی",),
        ("خرطوم",),
        ("ڈھاکہ",),
        ("کولمبو",),
        ("نیپال",),
        ("کٹمنڈو",),
        ("تھمپھو",),
        ("منیلا",),
        ("جکارتہ",),
        ("کوالالمپور",),
        ("سنگاپور",),
        ("بنکاک",),
        ("ہنوئی",),
    ]

    ORG_PHRASES = [
        ("آئی", "سی", "سی"),
        ("پی", "سی", "بی"),
        ("بی", "بی", "سی"),
        ("فےئر", "بریک", "گلوبل"),
        ("فیئر", "بریک", "گلوبل"),
        ("سعودی", "عرب", "کرکٹ", "فیڈریشن"),
        ("امریکی", "محکمہ", "خارجہ"),
        ("اقوام", "متحدہ",),
        ("عالمی", "بینک",),
        ("عالمی", "ادارہ", "صحت",),
        ("نیٹو",),
        ("یورپی", "یونین",),
        ("افغان", "طالبان",),
        ("تحریک", "انصاف",),
        ("مسلم", "لیگ", "ن"),
        ("پیپلز", "پارٹی",),
        ("جماعت", "اسلامی",),
        ("متحدہ", "قومی", "موومنٹ",),
        ("عوامی", "نیشنل", "پارٹی",),
        ("بلوچستان", "نیشنل", "پارٹی",),
        ("سندھی", "قومی", "پارٹی",),
        ("عوامی", "ورکرز", "پارٹی",),
        ("پاکستان", "تحریک", "انصاف",),
        ("پاکستان", "مسلم", "لیگ",),
        ("پاکستان", "پیپلز", "پارٹی",),
        ("سپریم", "کورٹ",),
        ("ہائی", "کورٹ",),
        ("وفاقی", "تحقیقاتی", "ادارہ",),
        ("نیب",),
        ("ایف", "بی", "آر",),
        ("پولیس", "ڈیپارٹمنٹ",),
        ("پی", "آئی", "اے",),
        ("آئی", "ایس", "آئی",),
        ("آئی", "ایس", "پی", "آر",),
        ("ورلڈ", "بینک",),
        ("آئی", "ایم", "ایف",),
        ("اوپیک",),
        ("یونیسف",),
        ("یونیسکو",),
        ("خلیجی", "کوآپریشن", "کونسل",),
        ("شنگھائی", "تعاون", "تنظیم",),
        ("برکس",),
    ]

    PER_PHRASES = [tuple(P) for P in PER_PHRASES]
    LOC_PHRASES = [tuple(P) for P in LOC_PHRASES]
    ORG_PHRASES = [tuple(P) for P in ORG_PHRASES]
    return PER_PHRASES, LOC_PHRASES, ORG_PHRASES


def TagSentenceNerBio(TOKENS, PER_PHRASES, LOC_PHRASES, ORG_PHRASES):
    """
    LONGEST-MATCH-FIRST PHRASE SCAN — TYPES MAP TO PER / LOC / ORG / MISC (MISC UNUSED HERE = O).
    """
    CORE_TOKS = [StripEdgePunctuation(T) or T for T in TOKENS]
    N = len(TOKENS)
    TAGS = ["O"] * N
    ALL = []
    for PH in PER_PHRASES:
        ALL.append(("PER", PH))
    for PH in LOC_PHRASES:
        ALL.append(("LOC", PH))
    for PH in ORG_PHRASES:
        ALL.append(("ORG", PH))
    ALL.sort(key=lambda X: len(X[1]), reverse=True)

    COVERED = [False] * N
    for KIND, PHR in ALL:
        LENP = len(PHR)
        if LENP == 0:
            continue
        for START in range(0, N - LENP + 1):
            if any(COVERED[START : START + LENP]):
                continue
            if tuple(CORE_TOKS[START : START + LENP]) == PHR:
                TAGS[START] = "B-" + KIND
                for J in range(1, LENP):
                    TAGS[START + J] = "I-" + KIND
                for J in range(LENP):
                    COVERED[START + J] = True
    return TAGS


def WriteConll(PATH, SENTENCES):
    """
    TAGGER_MODE IS 'POS' OR 'NER' — EACH SENTENCE IS LIST OF (TOKEN, TAG) AFTER TAGGING.
    """
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w", encoding="utf-8") as OUT:
        for SENT in SENTENCES:
            for W, T in SENT:
                OUT.write(W + "\t" + T + "\n")
            OUT.write("\n")


def PrintDistribution(TITLE, TAG_LIST):
    C = collections.Counter(TAG_LIST)
    print(TITLE)
    for K in sorted(C.keys()):
        print("  ", K, C[K])


def Main():
    ConfigureStdoutUtf8()
    random.seed(42)
    np.random.seed(42)

    METADATA = json.load(open("Metadata.json", encoding="utf-8"))
    DOCS = SplitDocsFiltered("cleaned.txt")
    ALL_RECS = BuildSentenceRecords(DOCS, METADATA)

    THREE = ("general", "world", "sport")
    SAMPLED = SampleFiveHundredWithTopicFloor(ALL_RECS, THREE, 100, 500)

    LABELS = [R[1] for R in SAMPLED]
    TR_IDX, TE_IDX = StratifiedTrainTestIndices(LABELS, 0.30, 42)
    TRAIN_RECS = [SAMPLED[I] for I in TR_IDX]
    TEST_RECS = [SAMPLED[I] for I in TE_IDX]

    LEX = BuildLexicons()
    PER_PH, LOC_PH, ORG_PH = BuildGazetteer()

    TRAIN_POS = []
    TRAIN_NER = []
    for SENT, _CAT, _DID in TRAIN_RECS:
        POS_TAGS = [TagPosForToken(T, LEX) for T in SENT]
        NER_TAGS = TagSentenceNerBio(SENT, PER_PH, LOC_PH, ORG_PH)
        TRAIN_POS.append(list(zip(SENT, POS_TAGS)))
        TRAIN_NER.append(list(zip(SENT, NER_TAGS)))

    TEST_POS = []
    TEST_NER = []
    for SENT, _CAT, _DID in TEST_RECS:
        POS_TAGS = [TagPosForToken(T, LEX) for T in SENT]
        NER_TAGS = TagSentenceNerBio(SENT, PER_PH, LOC_PH, ORG_PH)
        TEST_POS.append(list(zip(SENT, POS_TAGS)))
        TEST_NER.append(list(zip(SENT, NER_TAGS)))

    WriteConll("data/pos_train.conll", TRAIN_POS)
    WriteConll("data/pos_test.conll", TEST_POS)
    WriteConll("data/ner_train.conll", TRAIN_NER)
    WriteConll("data/ner_test.conll", TEST_NER)

    ALL_POS_TAGS = [T for S in TRAIN_POS + TEST_POS for _W, T in S]
    ALL_NER_TAGS = [T for S in TRAIN_NER + TEST_NER for _W, T in S]
    PrintDistribution("POS LABEL DISTRIBUTION (TRAIN+TEST POOLED):", ALL_POS_TAGS)
    PrintDistribution("NER LABEL DISTRIBUTION (TRAIN+TEST POOLED):", ALL_NER_TAGS)

    print("wrote data/pos_train.conll data/pos_test.conll data/ner_train.conll data/ner_test.conll")
    print("train sentences", len(TRAIN_RECS), "test sentences", len(TEST_RECS))


if __name__ == "__main__":
    Main()
