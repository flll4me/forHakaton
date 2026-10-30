from app.services.language_detector import detect_language
from app.services.normalizer import clean_text, fix_kazakh_cp1251, normalize_ticket, should_fix_kazakh


def test_cp1251_kazakh_table():
    assert fix_kazakh_cp1251("ќґѕєјўї") == "қғңәөұү"
    assert fix_kazakh_cp1251("ЌҐЅЄЈЎЇ") == "ҚҒҢӘӨҰҮ"


def test_cp1251_applied_on_kz_server_only_when_needed():
    n = normalize_ticket([{"role": "player", "text": "Ол мені балаѓаттады, ќорлады"}], region="kz")
    assert n.kazakh_fixed and "балағаттады" in n.clean_text and "қорлады" in n.clean_text
    assert n.original_text == "Ол мені балаѓаттады, ќорлады"  # исходный текст не меняется


def test_strong_substitute_triggers_fix_without_region():
    assert should_fix_kazakh("маѕан көмек керек")  # ѕ не встречается в ru/uk/uz


def test_ukrainian_and_uzbek_letters_not_converted():
    assert not should_fix_kazakh("поможіть мені будь ласка, перевернітє")  # украинские ї/є
    assert not should_fix_kazakh("ўзбекча матн", region="uz")  # узбекская ў


def test_clean_text_whitespace_and_control_chars():
    assert clean_text("  привет​ \n\n  мир\t ") == "привет мир"
    assert clean_text(None) == ""


def test_only_player_messages_used():
    n = normalize_ticket([
        {"role": "player", "text": "он меня оскорбил"},
        {"role": "reported_player", "text": "это была игра по роли"},
        {"role": "admin", "text": "Жалоба одобрена"},
    ])
    assert n.clean_text == "он меня оскорбил"
    assert n.accused_text == "это была игра по роли"


def test_empty_message():
    assert normalize_ticket([{"role": "player", "text": "  +  "}]).is_empty
    assert normalize_ticket([{"role": "player", "text": None}]).is_empty


def test_language_detection_core_languages():
    assert detect_language("он меня оскорбил без причины").lang == "ru"
    assert detect_language("Meni haqorat qildi onadan so'kdi").lang == "uz"
    assert detect_language("Мен ойыншы мені балағаттады, көмектесіңіз").lang == "kk"
    assert detect_language("he killed me for no reason please help").lang == "en"
    assert detect_language("+").lang == "und"
    assert detect_language("<url>").lang == "und"


def test_language_detection_ukrainian_maps_to_ru_and_slang():
    assert detect_language("поможіть мені будь ласка").lang == "ru"
    assert detect_language("дм").lang == "ru"            # кириллический сленг
    assert detect_language("DM <url>").lang == "und"     # только латинский сленг


def test_language_detection_typos_and_mixed():
    assert detect_language("памагите застрял в текстурах пжлст").lang == "ru"
    assert detect_language("admin iltimos yordam bering mashina tiqilib qoldi").lang == "uz"
