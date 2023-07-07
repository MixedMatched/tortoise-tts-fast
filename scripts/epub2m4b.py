from ebooklib import epub
import inflect
import spacy

import argparse
from html.parser import HTMLParser

import torch
import torchaudio

#from tortoise.api import MODELS_DIR, TextToSpeech
#from tortoise.utils.audio import load_audio
#from tortoise.utils.diffusion import SAMPLERS
#from tortoise.models.vocoder import VocConf

months = ["January", "February", "March", 
            "April", "May", "June", 
            "July", "August", "September",
            "October", "November", "December"]

def is_number(word):
    for char in word:
        if not (char.isdigit() or char == "." or char == ","):
            return False
    return True

def replace_numbers_with_words(text):
    p = inflect.engine()
    words = text.split()
    for i in range(len(words)):
        word = words[i]

        post_punctuation = ""
        while len(word) > 0 and not word[-1].isalnum():
            post_punctuation += word[-1]
            word = word[:-1]

        post_punctuation = post_punctuation[::-1]

        if len(word) == 0:
            continue

        pre_punctuation = ""
        while not word[0].isalnum():
            pre_punctuation += word[0]
            word = word[1:]
        
        if word.isdigit(): # number with just digits
            if len(word) == 4: # year
                words[i] = ' '.join(p.number_to_words(word, group=2, wantlist=True))
            elif (len(word) == 1 or len(word) == 2) and i > 0 and words[i - 1] in months: # day of month
                words[i] = p.ordinal(p.number_to_words(word))
            else: # number
                words[i] = p.number_to_words(word)
        elif word.endswith("s") and word[:-1].isdigit(): # number with s
            number = word[:-1]
            if len(number) == 4: # group of years
                nonplural = ' '.join(p.number_to_words(number, group=2, wantlist=True))
                plural = p.plural(nonplural)
                words[i] = plural
            else: 
                words[i] = p.number_to_words(number) + "s"
        elif (word.endswith("th") or word.endswith("st") or word.endswith("nd") or word.endswith("rd")) and word[:-2].isdigit(): # ordinal number
            number = word[:-2]
            words[i] = p.ordinal(p.number_to_words(number))
        elif word[:-1].isdigit():
            number = word[:-1]
            if len(number) == 4:
                words[i] = ' '.join(p.number_to_words(number, group=2, wantlist=True)) + word[-1]
            else:
                words[i] = p.number_to_words(number) + word[-1]
        elif word.startswith("$") and is_number(word[1:]): 
            number = word[1:]
            words[i] = p.number_to_words(number, decimal = "point") + " dollar" + ("s" if number != "1" else "")
        elif is_number(word):
            words[i] = p.number_to_words(word, decimal = "point")
        else:
            num = ""
            new_word = ""
            for j in range(len(word)):
                if word[j].isdigit():
                    num += word[j]
                else:
                    new_word += word[j]
                    if num != "":
                        new_word += p.number_to_words(num)
                        num = ""                
                    break
            continue
        words[i] += post_punctuation
        words[i] = pre_punctuation + words[i]
    return ' '.join(words)

def remove_periods(text, nlp): 
    # Process the text
    doc = nlp(text)
    
    # Go through every word in the text
    new_text = ""
    for token in doc:
        # If word is a proper noun and not the last word in the text
        if token.is_title and token.i<len(doc) - 1: 
            new_text += token.text_with_ws.replace(".", "")
        # Else keep the word as it is
        else:
            new_text += token.text_with_ws
    return new_text

def expand_acronyms(text):
    words = text.split()

    for i in range(len(words)):
        word = words[i]

        post_punctuation = ""
        while len(word) > 0 and not word[-1].isalnum():
            post_punctuation += word[-1]
            word = word[:-1]

        post_punctuation = post_punctuation[::-1]

        if len(word) == 0:
            continue

        pre_punctuation = ""
        while not word[0].isalnum():
            pre_punctuation += word[0]
            word = word[1:]

        if (i > 0 and words[i-1].isupper()) or (i < len(words) - 1 and words[i+1].isupper()):
            continue
        elif word.isupper() and len(word) > 1:
            words[i] = ' '.join(word)
        elif word[-1] == "s" and word[:-1].isupper() and len(word) > 2:
            words[i] = ' '.join(word[:-1]) + "s"
        else:
            continue

        words[i] += post_punctuation
        words[i] = pre_punctuation + words[i]
    return ' '.join(words)

def process_text(text, nlp):
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("—", ", ")
    text = text.replace("–", ", ")
    text = text.replace("--", ", ")
    text = text.replace("-", " ")
    text = text.replace("…", "...")
    text = text.replace("=", " equals ")
    text = text.replace("+", " plus ")
    text = text.replace("-", " minus ")

    text = text.replace("Mr.", "Mister")
    text = text.replace("Mrs.", "Missus")
    text = text.replace("Ms.", "Miss")
    text = text.replace("Dr.", "Doctor")
    text = text.replace("Prof.", "Professor")
    text = text.replace("St.", "Saint")
    text = text.replace("Mt.", "Mount")
    text = text.replace("Ft.", "Fort")

    text = replace_numbers_with_words(text)
    text = text.replace("-", " ")
    text = remove_periods(text, nlp)
    text = expand_acronyms(text)

    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert epub to m4b')
    parser.add_argument('epub', metavar='epub', type=str, nargs='+',
                        help='path to epub file')
    
    args = parser.parse_args()

    book = epub.read_epub(args.epub[0])
    
    items = []
    print("Enter the numbers of the chapter(s) you want to convert to m4b:")
    for item in book.items:
        if isinstance(item, epub.EpubHtml):
            items.append(item)
            print(len(items), ": ", item.get_name())

    chapters = input("Enter the chapter numbers separated by dashes for a range or commas: ")
    chapters = chapters.split(",")

    chapter_indices = []
    for chapter in chapters:
        if "-" in chapter:
            chapter_range = chapter.split("-")
            chapter_range = range(int(chapter_range[0]) - 1, int(chapter_range[1]))
            for i in chapter_range:
                chapter_indices.append(i)
        else:
            chapter_indices.append(int(chapter) - 1)

    audio_by_chapter = []
    current_chapter = 0

    nlp = spacy.load("en_core_web_sm")

    class EpubHTMLParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.current_text = ""
        def handle_starttag(self, tag, attrs):
            if tag == "p":
                ...
        def handle_data(self, data):
            data = process_text(data, nlp)
            self.current_text += data

    for (i, item) in enumerate(items):
        if isinstance(item, epub.EpubHtml) and i in chapter_indices:
            print(item.get_name())
            parser = EpubHTMLParser()
            parser.feed(item.get_body_content().decode("utf-8"))