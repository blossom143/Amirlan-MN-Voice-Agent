"""
Khan Bank Colloquial & Spoken Mongolian Banking Glossary
=========================================================
Maps colloquial, everyday spoken Mongolian expressions to official banking terminology.
Provides bidirectional synonym matching and query expansion for hybrid search and reranking.
"""

from typing import Dict, List, Set, Tuple, Any
import re

# Comprehensive Colloquial to Official Banking Term Mapping
MONGOLIAN_BANKING_GLOSSARY: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # LOANS (ЗЭЭЛ)
    # -------------------------------------------------------------------------
    "байрны зээл": {
        "canonical": "Орон сууцны ипотекийн 6% зээл",
        "category": "loan_inquiry",
        "keywords": ["ипотек", "6 хувь", "урьдчилгаа 30%", "100 м.кв", "30 жил", "НДШ 12 сар"],
        "url": "https://www.khanbank.com/mn/personal/loans/mortgage-loan"
    },
    "орон сууцны зээл": {
        "canonical": "Орон сууцны ипотекийн 6% зээл",
        "category": "loan_inquiry",
        "keywords": ["ипотек", "хөнгөлөлттэй зээл", "журам", "урьдчилгаа 30%"],
        "url": "https://www.khanbank.com/mn/personal/loans/mortgage-loan"
    },
    "хөдөөний байрны зээл": {
        "canonical": "Хөдөө орон нутгийн 3% ипотекийн зээл",
        "category": "loan_inquiry",
        "keywords": ["орон нутаг", "3 хувь", "шилжин суурьших", "хөнгөлөлттэй хүү"],
        "url": "https://www.khanbank.com/mn/personal/loans/mortgage-loan"
    },
    "цалингийн зээл": {
        "canonical": "Khan Bank App цалингийн зээл (Цахим зээл)",
        "category": "loan_inquiry",
        "keywords": ["аппын зээл", "шимтгэл 0%", "50 сая төгрөг", "30 сар", "урьдчилан баталгаажсан"],
        "url": "https://www.khanbank.com/mn/personal/loans/salary-loan"
    },
    "аппын зээл": {
        "canonical": "Цахим цалингийн болон хэрэглээний зээл",
        "category": "loan_inquiry",
        "keywords": ["онлайн зээл", "утсаар зээл авах", "Khan Bank App", "0 төгрөг шимтгэл"],
        "url": "https://www.khanbank.com/mn/personal/loans/salary-loan"
    },
    "машины зээл": {
        "canonical": "Автомашины хэрэглээний зээл",
        "category": "loan_inquiry",
        "keywords": ["авто зээл", "шинэ машин", "хуучин машин", "урьдчилгаа 20-30%", "60 сар"],
        "url": "https://www.khanbank.com/mn/personal/loans/auto-loan"
    },
    "тэтгэврийн зээл": {
        "canonical": "Тэтгэврийн зээл болон урьдчилгаа",
        "category": "loan_inquiry",
        "keywords": ["өндөр настан", "тэтгэвэр барьцаалсан", "хөнгөлөлттэй"],
        "url": "https://www.khanbank.com/mn/personal/loans/pension-loan"
    },
    "малчны зээл": {
        "canonical": "Малчны зээл",
        "category": "loan_inquiry",
        "keywords": ["ноолуур", "мал сүрэг", "эргэн төлөлт уян хатан"],
        "url": "https://www.khanbank.com/mn/personal/loans/herder-loan"
    },

    # -------------------------------------------------------------------------
    # CARDS & ISSUANCE (КАРТ БОЛОН АТМ)
    # -------------------------------------------------------------------------
    "карт залгисан": {
        "canonical": "АТМ-д карт саатуулагдсан үеийн журам",
        "category": "troubleshooting",
        "keywords": ["карт гацсан", "атм авчихлаа", "түр түгжих", "3 ажлын өдөр", "салбараас авах"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт залгичихлаа": {
        "canonical": "АТМ-д карт саатуулагдсан үеийн журам",
        "category": "troubleshooting",
        "keywords": ["карт гацсан", "атм авчихлаа", "түр түгжих", "3 ажлын өдөр", "салбараас авах"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт гацсан": {
        "canonical": "АТМ-д карт саатуулагдсан үеийн журам",
        "category": "troubleshooting",
        "keywords": ["карт түгжих", "1800-1917", "киоскоор шинээр авах", "10,000₮"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт гацчихлаа": {
        "canonical": "АТМ-д карт саатуулагдсан үеийн журам",
        "category": "troubleshooting",
        "keywords": ["карт түгжих", "1800-1917", "киоскоор шинээр авах", "10,000₮"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт хаясан": {
        "canonical": "Карт гээгдүүлсэн үед яаралтай хаах, дахин авах",
        "category": "troubleshooting",
        "keywords": ["карт гээсэн", "түр түгжих", "киоскоор 1 минутад хэвлэх", "10,000₮"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт хаячихлаа": {
        "canonical": "Карт гээгдүүлсэн үед яаралтай хаах, дахин авах",
        "category": "troubleshooting",
        "keywords": ["карт гээсэн", "түр түгжих", "киоскоор 1 минутад хэвлэх", "10,000₮"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт гээчихлээ": {
        "canonical": "Карт гээгдүүлсэн үед яаралтай хаах, дахин авах",
        "category": "troubleshooting",
        "keywords": ["карт гээсэн", "түр түгжих", "киоскоор 1 минутад хэвлэх", "10,000₮"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт хулгайд алдсан": {
        "canonical": "Карт яаралтай хаах, аюулгүй байдал",
        "category": "troubleshooting",
        "keywords": ["яаралтай түгжих", "1800-1917", "App картууд цэс", "гүйлгээ хаах"],
        "url": "https://www.khanbank.com/mn/personal/contact"
    },
    "код мартсан": {
        "canonical": "Картын ПИН код болон Цахим банкны нэвтрэх код сэргээх",
        "category": "troubleshooting",
        "keywords": ["пин код солих", "3 удаа буруу оруулсан", "аппаар пин шинэчлэх", "үнэгүй"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "код мартчихлаа": {
        "canonical": "Картын ПИН код болон Цахим банкны нэвтрэх код сэргээх",
        "category": "troubleshooting",
        "keywords": ["пин код солих", "3 удаа буруу оруулсан", "аппаар пин шинэчлэх", "үнэгүй"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "түгжигдчихлээ": {
        "canonical": "Картын ПИН код 3 удаа буруу оруулсан түгжээ тайлах",
        "category": "troubleshooting",
        "keywords": ["түгжээ тайлах", "аппаар нээх", "ПИН сэргээх", "Картууд цэс"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "пин код": {
        "canonical": "Картын ПИН код тохируулах, сэргээх",
        "category": "card_inquiry",
        "keywords": ["PIN код", "нууц үг", "буруу оруулсан түгжээ тайлах", "Киоск"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "карт хэвлэх": {
        "canonical": "Дижитал Киоскоор 1 минутад карт хэвлэх",
        "category": "card_inquiry",
        "keywords": ["Экспресс киоск", "10,000₮ шимтгэл", "дебит карт", "хурууны хээ", "24/7"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "ногоон карт": {
        "canonical": "Үндэсний төлбөрийн дебит карт (₮ карт)",
        "category": "card_inquiry",
        "keywords": ["үндэсний карт", "₮ карт", "жилийн хураамж 5,000₮", "дотоодын төлбөр тооцоо"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },
    "виза карт": {
        "canonical": "Олон улсын Visa Classic, Gold, Platinum, Signature картууд",
        "category": "card_inquiry",
        "keywords": ["Visa", "Mastercard", "гадаадад уншуулах", "онлайн худалдан авалт", "Lounge Key"],
        "url": "https://www.khanbank.com/mn/personal/cards"
    },

    # -------------------------------------------------------------------------
    # DIGITAL BANKING & LIMITS (ЦАХИМ БАНК БОЛОН ЛИМИТ)
    # -------------------------------------------------------------------------
    "гүйлгээний хязгаар": {
        "canonical": "Khan Bank App гүйлгээний өдрийн лимит ба хязгаар",
        "category": "digital_banking",
        "keywords": ["SMS OTP 20 сая", "Smart Tan 50 сая", "Тоон гарын үсэг 100 сая", "хязгаар өсгөх"],
        "url": "https://www.khanbank.com/mn/personal/digital-banking"
    },
    "лимит": {
        "canonical": "Khan Bank App гүйлгээний лимит",
        "category": "digital_banking",
        "keywords": ["гүйлгээний дээд хэмжээ", "20 сая", "50 сая", "100 сая төгрөг"],
        "url": "https://www.khanbank.com/mn/personal/digital-banking"
    },
    "данс нээх": {
        "canonical": "Khan Bank App-аар цахимаар данс нээх",
        "category": "account_inquiry",
        "keywords": ["E-Mongolia царай таних", "0₮ шимтгэл", "салбарт очихгүй", "харилцах данс"],
        "url": "https://www.khanbank.com/mn/personal/accounts"
    },
    "дансны хуулга": {
        "canonical": "Дансны хуулга болон тодорхойлолт авах",
        "category": "account_inquiry",
        "keywords": ["тамгатай хуулга", "Киоскоор хэвлэх", "Khan Bank App pdf татах", "500₮-1,000₮"],
        "url": "https://www.khanbank.com/mn/personal/accounts"
    },
    "гүйлгээний шимтгэл": {
        "canonical": "Шилжүүлгийн шимтгэл ба тарифын матриц",
        "category": "fees_and_tariffs",
        "keywords": ["Хаан дотор 0₮", "Бусад банк 5 саяас доош 100₮", "5 саяас дээш 500₮ RTGS"],
        "url": "https://www.khanbank.com/mn/personal/fees-and-tariffs"
    },
    "өөр банк": {
        "canonical": "Банк хооронд ба өөр банк руу шилжүүлэх шимтгэл",
        "category": "fees_and_tariffs",
        "keywords": ["бусад банк", "өөр банк руу шилжүүлэх", "100₮", "500₮ RTGS"],
        "url": "https://www.khanbank.com/mn/personal/tariffs"
    },
    "өөр банк руу": {
        "canonical": "Банк хооронд ба өөр банк руу шилжүүлэх шимтгэл",
        "category": "fees_and_tariffs",
        "keywords": ["бусад банк", "өөр банк руу шилжүүлэх", "100₮", "500₮ RTGS"],
        "url": "https://www.khanbank.com/mn/personal/tariffs"
    },
    "код ирэхгүй байна": {
        "canonical": "Нэг удаагийн нууц үг (SMS OTP) хүлээн авах заавар",
        "category": "troubleshooting",
        "keywords": ["OTP код ирэхгүй", "мессеж ирэхгүй", "сүлжээ шалгах", "лавлах 1800-1917"],
        "url": "https://www.khanbank.com/mn/personal/troubleshooting"
    },
    "мөнгө орсонгүй": {
        "canonical": "Банк хоорондын гүйлгээний саатал, клирингийн мэдээлэл",
        "category": "troubleshooting",
        "keywords": ["мөнгө гацсан", "гүйлгээ саатсан", "бага дүн клиринг", "RTGS систем"],
        "url": "https://www.khanbank.com/mn/personal/troubleshooting"
    },

    # -------------------------------------------------------------------------
    # SAVINGS (ХАДГАЛАМЖ)
    # -------------------------------------------------------------------------
    "хадгаламжийн хүү": {
        "canonical": "Иргэдийн хугацаатай болон хугацаагүй хадгаламжийн хүү",
        "category": "deposit_inquiry",
        "keywords": ["хугацаатай 9.6%-14.4%", "хугацаагүй 3.6%-4.8%", "хүүгээ сар бүр авах"],
        "url": "https://www.khanbank.com/mn/personal/savings"
    },
    "хүүхдийн хадгаламж": {
        "canonical": "Хүүхдийн ирээдүйн хадгаламж",
        "category": "deposit_inquiry",
        "keywords": ["жилийн хүү 13.2%-14.0%", "хүүхдийн мөнгө холбох", "капиталжих"],
        "url": "https://www.khanbank.com/mn/personal/savings"
    },
    "хадгаламжийн даатгал": {
        "canonical": "Хадгаламжийн даатгалын корпораци (ДИКОМ) баталгаа",
        "category": "deposit_inquiry",
        "keywords": ["20 сая төгрөг хүртэл", "хуулиар даатгагдсан", "найдвартай"],
        "url": "https://www.khanbank.com/mn/personal/savings"
    },

    # -------------------------------------------------------------------------
    # BRANCHES, ATMS & WORKING HOURS (САЛБАР, АЖИЛЛАХ ЦАГ)
    # -------------------------------------------------------------------------
    "хагас сайнд ажиллах салбар": {
        "canonical": "Бямба гарагт болон амралтын өдрүүдэд ажиллах салбаруудын хуваарь",
        "category": "branch_and_atm",
        "keywords": ["Бямба 10:00-16:00", "Улаанбаатар салбар", "Залуус", "Сансар", "Түмэн шувуут", "Дархан", "Эрдэнэт"],
        "url": "https://www.khanbank.com/mn/personal/branches-and-atms"
    },
    "бямба гарагт ажиллах салбар": {
        "canonical": "Бямба гарагт болон амралтын өдрүүдэд ажиллах салбаруудын хуваарь",
        "category": "branch_and_atm",
        "keywords": ["Бямба гараг", "Улаанбаатар төв салбар", "Залуус салбар", "Сансар", "Түмэн шувуут"],
        "url": "https://www.khanbank.com/mn/personal/branches-and-atms"
    },
    "бүтэн сайнд ажиллах салбар": {
        "canonical": "Ням гарагт ажиллах Шангри-Ла салбар ба 24/7 Экспресс төв",
        "category": "branch_and_atm",
        "keywords": ["Шангри-Ла салбар 10:00-20:00", "24/7 Экспресс төвүүд", "Дижитал Киоск"],
        "url": "https://www.khanbank.com/mn/personal/branches-and-atms"
    },
    "орой ажиллах салбар": {
        "canonical": "Уртасгасан цагаар ажиллах салбарууд",
        "category": "branch_and_atm",
        "keywords": ["Шангри-Ла 20:00 хүртэл", "Зайсан Хилл 19:30 хүртэл", "Экспресс 24/7"],
        "url": "https://www.khanbank.com/mn/personal/branches-and-atms"
    },
    "24 цагийн салбар": {
        "canonical": "24/7 Экспресс банкны төв болон Дижитал Киоск",
        "category": "branch_and_atm",
        "keywords": ["24 цаг", "карт шууд хэвлэх", "ЦДМ АТМ бэлэн мөнгө тушаах", "хуулга авах"],
        "url": "https://www.khanbank.com/mn/personal/branches-and-atms"
    },
    "мөнгө тушаах атм": {
        "canonical": "Бэлэн мөнгө хүлээн авах ЦДМ АТМ сүлжээ",
        "category": "branch_and_atm",
        "keywords": ["ЦДМ АТМ", "бэлэн мөнгө хийх", "500 гаруй АТМ", "24 цагаар дансанд шууд орох"],
        "url": "https://www.khanbank.com/mn/personal/branches-and-atms"
    },

    # -------------------------------------------------------------------------
    # REMITTANCE, FOREX & SWIFT (ГАДААД ГҮЙВҮҮЛГЭ, SWIFT, ВАЛЮТ)
    # -------------------------------------------------------------------------
    "свифт код": {
        "canonical": "Хаан Банкны SWIFT / BIC код KHANMNUB",
        "category": "remittance_and_forex",
        "keywords": ["SWIFT BIC KHANMNUB", "гадаадаас мөнгө авах", "Khan Bank LLC"],
        "url": "https://www.khanbank.com/mn/personal/remittance/international-transfer"
    },
    "swift": {
        "canonical": "Хаан Банкны SWIFT / BIC код KHANMNUB",
        "category": "remittance_and_forex",
        "keywords": ["KHANMNUB", "SWIFT code", "International Wire Transfer"],
        "url": "https://www.khanbank.com/mn/personal/remittance/international-transfer"
    },
    "гадаадаас мөнгө авах": {
        "canonical": "Гадаад улсаас SWIFT гүйлгээ хүлээн авах заавар",
        "category": "remittance_and_forex",
        "keywords": ["KHANMNUB", "10 оронтой данс", "латин нэр", "валютын данс"],
        "url": "https://www.khanbank.com/mn/personal/remittance/international-transfer"
    },
    "вестерн юнион": {
        "canonical": "Western Union шуурхай олон улсын гуйвуулга",
        "category": "remittance_and_forex",
        "keywords": ["хүлээн авах шимтгэл 0₮", "Khan Bank App-аар шууд авах", "10 оронтой MTCN"],
        "url": "https://www.khanbank.com/mn/personal/remittance/international-transfer"
    },
    "western union": {
        "canonical": "Western Union шуурхай гуйвуулга",
        "category": "remittance_and_forex",
        "keywords": ["MTCN дугаар", "аппаар хүлээн авах", "0₮ шимтгэл", "салбараас бэлнээр авах"],
        "url": "https://www.khanbank.com/mn/personal/remittance/international-transfer"
    },
    "валют солих": {
        "canonical": "Khan Bank App болон салбараар валют арилжаа хийх",
        "category": "remittance_and_forex",
        "keywords": ["ханш", "USD", "EUR", "CNY", "JPY", "цахимаар хөрвүүлэх"],
        "url": "https://www.khanbank.com/mn/personal/remittance/international-transfer"
    },

    # -------------------------------------------------------------------------
    # CONTACT, COMPLAINTS & ESCALATION (ХОЛБОГДОХ, ЛАВЛАХ, ЭСКАЛАЦИ)
    # -------------------------------------------------------------------------
    "лавлах": {
        "canonical": "24/7 Харилцагчийн лавлах төв 1800-1917",
        "category": "contact_and_support",
        "keywords": ["1800-1917", "гадаадаас +976 7515-1917", "24 цаг тасралтгүй"],
        "url": "https://www.khanbank.com/mn/personal/contact"
    },
    "утас": {
        "canonical": "Хаан Банкны харилцагчийн лавлах төв 1800-1917",
        "category": "contact_and_support",
        "keywords": ["1800-1917", "лавлах утас", "байгууллага +976 7515-3333"],
        "url": "https://www.khanbank.com/mn/personal/contact"
    },
    "хүнтэй ярих": {
        "canonical": "Лавлах төвийн оператортай холбогдох (Эскалаци)",
        "category": "contact_and_support",
        "keywords": ["операторт шилжүүлэх", "1800-1917", "амьд мэргэжилтэн"],
        "url": "https://www.khanbank.com/mn/personal/contact"
    },
    "оператор": {
        "canonical": "Лавлах төвийн амьд оператортой холбогдох (1800-1917)",
        "category": "contact_and_support",
        "keywords": ["амьд холболт", "1800-1917", "хувийн данс тооцоо нийлэх"],
        "url": "https://www.khanbank.com/mn/personal/contact"
    }
}


def expand_query_with_banking_terms(query: str) -> str:
    """
    Expands a colloquial Mongolian user query by appending canonical banking keywords.
    Ensures that queries like 'байрны зээл хэдэн хувь бэ' match documents containing
    'Орон сууцны ипотекийн 6% зээл'.
    """
    if not query:
        return ""

    q_lower = query.lower()
    expanded_additions: List[str] = []
    matched_categories: Set[str] = set()

    for slang_term, entry in MONGOLIAN_BANKING_GLOSSARY.items():
        if slang_term in q_lower:
            canonical = entry["canonical"]
            keywords = " ".join(entry.get("keywords", []))
            expanded_additions.append(f"{canonical} {keywords}")
            matched_categories.add(entry.get("category", "general"))

    if not expanded_additions:
        return query

    # Combine original query with boosted canonical expansions
    combined = f"{query} {' '.join(expanded_additions)}"
    # Deduplicate whitespace
    return re.sub(r"\s+", " ", combined).strip()


def extract_matched_glossary_metadata(query: str) -> List[Dict[str, Any]]:
    """
    Returns matched glossary entries for a given query to enhance retrieval reranking.
    """
    if not query:
        return []

    q_lower = query.lower()
    matches = []
    for term, data in MONGOLIAN_BANKING_GLOSSARY.items():
        if term in q_lower:
            matches.append({
                "slang_term": term,
                "canonical": data["canonical"],
                "category": data["category"],
                "url": data.get("url", "https://www.khanbank.com"),
                "keywords": data.get("keywords", [])
            })
    return matches
