import re

# Symbol and abbreviation mappings with space after symbols
symbol_map = {
    "+": "ప్లస్ ",
    "-": "మైనస్ ",
    "%": "శాతం ",
    "&": "మరియు "
}

abbrev_map = {
    "etc.": "ఇతరులు ",
    "e.g.": "ఉదాహరణకి ",
    "i.e.": "అంటే ",
    "vs": "వర్సెస్ ",
    "Dept.": "విభాగం ",
    "Govt.": "ప్రభుత్వం ",
    "Pvt. Ltd.": "ప్రైవేట్ లిమిటెడ్ ",
    "Ltd.": "లిమిటెడ్ ",
    "Mr.": "శ్రీ ",
    "Mrs.": "శ్రీమతి ",
    "Dr.": "డాక్టర్ ",
    "Prof.": "ప్రొఫెసర్ ",
    "Hon.": "గౌరవనీయులు "
}

# Telugu number map
telugu_digits = {
    '0': 'సున్నా',
    '1': 'ఒకటి',
    '2': 'రెండు',
    '3': 'మూడు',
    '4': 'నాలుగు',
    '5': 'ఐదు',
    '6': 'ఆరు',
    '7': 'ఏడు',
    '8': 'ఎనిమిది',
    '9': 'తొమ్మిది'
}

def number_to_telugu(n: int) -> str:
    if n == 100:
        return "ఒక వంద"
    if 1 <= n < 10:
        return telugu_digits[str(n)]
    elif n < 20:
        special = {
            10: "పది", 11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు",
            14: "పద్నాలుగు", 15: "పదహేను", 16: "పదహారు", 17: "పదిహేడు",
            18: "పద్దెనిమిది", 19: "పంతొమ్మిది"
        }
        return special[n]
    elif n < 100:
        tens = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్బై", "ఎనభై", "తొంభై"]
        ones = ["", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది"]
        return tens[n // 10] + (" " + ones[n % 10] if n % 10 else "")
    elif n < 1000:
        hundreds_place = n // 100
        if hundreds_place == 1:
            return "ఒక వంద" + (" " + number_to_telugu(n % 100) if n % 100 else "")
        else:
            return telugu_digits[str(hundreds_place)] + " వందలు" + (" " + number_to_telugu(n % 100) if n % 100 else "")
    elif n < 2000:
        # For numbers below 2000, no 'ఒకటి వేల'. Just handle as hundreds and tens
        return number_to_telugu(n // 100) + " వందలు" + (" " + number_to_telugu(n % 100) if n % 100 else "")
    else:
        return number_to_telugu(n // 1000) + " వేల" + (" " + number_to_telugu(n % 1000) if n % 1000 else "")

# Special handling for years
def year_to_telugu(n: int) -> str:
    if 1900 <= n <= 1999:
        return "పంతొమ్మిది వందల " + number_to_telugu(n % 100) + " సంవత్సరం"
    elif 2000 <= n <= 2099:
        return "రెండు వేల " + number_to_telugu(n % 100) + " సంవత్సరం"
    else:
        return number_to_telugu(n) + " సంవత్సరంము"  # fallback

# Special handling for ordinal (1st, 2nd, 3rd)
def get_ordinal(n: int) -> str:
    ordinals = {
        1: "ఒకటవ",
        2: "రెండవ",
        3: "మూడవ",
        4: "నాలుగవ",
        5: "ఐదవ",
        6: "ఆరవ",
        7: "ఏడవ",
        8: "ఎనిమిదవ",
        9: "తొమ్మిదవ",
        10: "పదవ",
    }
    return ordinals.get(n, f"{number_to_telugu(n)}వ")

# Currency replacers
def dollar_repl(match):
    n = int(match.group(1))
    return "ఒక డాలరు" if n == 1 else number_to_telugu(n) + " డాలర్లు"

def rupee_repl(match):
    n = int(match.group(1))
    return "ఒక రూపాయి" if n == 1 else number_to_telugu(n) + " రూపాయలు"

def euro_repl(match):
    n = int(match.group(1))
    return "ఒక యూరో" if n == 1 else number_to_telugu(n) + " యూరోలు"

def pound_repl(match):
    n = int(match.group(1))
    return "ఒక పౌండ్" if n == 1 else number_to_telugu(n) + " పౌండ్లు"

def percent_repl(match):
    n = int(match.group(1))
    return number_to_telugu(n) + " శాతం"

# Date replacer
def replace_dates(text):
    date_pattern = r'(\b\d{1,2})[/-](\d{1,2})[/-](\d{2,4})'
    def date_repl(match):
        d, m, y = map(int, match.groups())
        year_in_telugu = year_to_telugu(y)  # Special handling for year
        return f"{get_ordinal(d)} తేదీ {get_ordinal(m)} నెల {year_in_telugu}"
    return re.sub(date_pattern, date_repl, text)

# General number replacer
def replace_numbers(text):
    def number_repl(match):
        num = int(match.group())
        return number_to_telugu(num)
    return re.sub(r'\b\d{1,4}\b', number_repl, text)

# Full normalization
def normalize_text(text):
    # Replace symbols with a space after them
    for sym, rep in symbol_map.items():
        text = text.replace(sym, f" {rep}")

    # Replace abbreviations with a space after them
    for abbr, full in abbrev_map.items():
        text = text.replace(abbr, full)

    # Handle date and number normalization
    text = replace_dates(text)
    text = re.sub(r'₹+', '₹', text)  # Remove repeated ₹
    text = re.sub(r'\$(\d+)', dollar_repl, text)
    text = re.sub(r'₹(\d+)', rupee_repl, text)
    text = re.sub(r'€(\d+)', euro_repl, text)
    text = re.sub(r'£(\d+)', pound_repl, text)
    text = re.sub(r'(\d+)%', percent_repl, text)

    text = replace_numbers(text)
    return text.strip()

# Example usage
text = "Dr.మనోహర్ 01/5/1999 మార్కెట్ కి వెళ్లి  $5 టమోటాలు & ₹25 చికెన్ కొన్నాడు. మొత్తం ₹1921 బిల్లు అయ్యింది. 2-5-2025 అతనికి  10% డిస్కౌంట్ వచ్చింది."
print(normalize_text(text))
