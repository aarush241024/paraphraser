import streamlit as st
import openai
from dotenv import load_dotenv
import os
from langdetect import detect, LangDetectException
import difflib
import nltk
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set word limit
WORD_LIMIT = 250

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # Default to English if detection fails

def get_full_language_name(lang_code):
    language_dict = {
        'en': 'English',
        'zh-cn': 'Chinese',
        'hi': 'Hindi',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'ja': 'Japanese',
        'ru': 'Russian',
        'it': 'Italian',
        'ko': 'Korean',
        'he': 'Hebrew',
        'ar': 'Arabic',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'fi': 'Finnish',
        'cs': 'Czech',
        'da': 'Danish',
        'sv': 'Swedish',
        'pl': 'Polish',
        'ro': 'Romanian',
        'vi': 'Vietnamese',
        'id': 'Indonesian',
        'tr': 'Turkish',
        'el': 'Greek',
        'no': 'Norwegian',
        'hu': 'Hungarian',
        'sk': 'Slovak',
        'et': 'Estonian',
        'lt': 'Lithuanian',
        'gu': 'Gujarati',
        'bn': 'Bengali',
        'kn': 'Kannada',
        'ta': 'Tamil',
        'te': 'Telugu'
    }
    return language_dict.get(lang_code, 'Unknown')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def get_synonyms(word):
    lemmatizer = WordNetLemmatizer()
    
    # Get the part of speech
    pos = pos_tag([word])[0][1]
    wordnet_pos = get_wordnet_pos(pos)
    
    # Lemmatize the word
    lemma = lemmatizer.lemmatize(word, wordnet_pos)
    
    synonyms = set()
    for syn in wordnet.synsets(lemma):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if wordnet_pos == wordnet.VERB:
                if pos.startswith('VB'):
                    synonym = lemmatizer.lemmatize(synonym, wordnet.VERB)
                if pos == 'VBD':  # Past tense
                    synonym = lemmatizer.lemmatize(synonym, wordnet.VERB) + 'ed'
                elif pos == 'VBG':  # Gerund/present participle
                    synonym = lemmatizer.lemmatize(synonym, wordnet.VERB) + 'ing'
                elif pos == 'VBN':  # Past participle
                    synonym = lemmatizer.lemmatize(synonym, wordnet.VERB) + 'ed'
                elif pos == 'VBP' or pos == 'VBZ':  # Non-3rd person singular present, or 3rd person singular present
                    synonym = lemmatizer.lemmatize(synonym, wordnet.VERB)
            synonyms.add(synonym)
    
    return list(synonyms)[:5]  # Limit to 5 synonyms

def paraphrase_and_translate(text, input_language, output_language, mode, synonym_level, expand_shorten_option, custom_instructions, paraphrase_quotations, avoid_contractions, prefer_active_voice):
    prompt = f"""Translate and paraphrase the following text from {input_language} to {output_language}.
    Use the {mode} style and apply a synonym level of {synonym_level}/5.
    {"Expand the text for " + expand_shorten_option + "." if mode == "Expand" else ""}
    {"Shorten the text for " + expand_shorten_option + "." if mode == "Shorten" else ""}
    Custom instructions: {custom_instructions}
    {"Paraphrase quotations." if paraphrase_quotations else "Do not paraphrase quotations."}
    {"Avoid contractions." if avoid_contractions else ""}
    {"Prefer active voice." if prefer_active_voice else ""}
    Provide a single, coherent paraphrased version in {output_language}:

    {text}
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that translates and paraphrases text."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message['content'].strip()

def highlight_differences_with_synonyms(original, paraphrased, use_yellow_highlight, show_changed_words, show_longest_unchanged_words):
    original_sentences = original.split('.')
    paraphrased_sentences = paraphrased.split('.')
    
    highlighted_sentences = []
    
    for orig_sent, para_sent in zip(original_sentences, paraphrased_sentences):
        orig_words = orig_sent.split()
        para_words = para_sent.split()
        
        d = difflib.Differ()
        diff = list(d.compare(orig_words, para_words))
        
        highlighted_words = []
        unchanged_sequence = []
        longest_unchanged_sequence = []

        for word in diff:
            if word.startswith('  '):
                unchanged_sequence.append(word[2:])
                if len(unchanged_sequence) > len(longest_unchanged_sequence):
                    longest_unchanged_sequence = unchanged_sequence.copy()
                highlighted_words.append(f'<span>{word[2:]}</span>')
            elif word.startswith('+ '):
                unchanged_sequence = []
                synonyms = get_synonyms(word[2:])
                synonym_str = ",".join(synonyms) if synonyms else ""
                if show_changed_words:
                    highlighted_words.append(f'<span class="changed-word tooltip" data-word="{word[2:]}" data-synonyms="{synonym_str}">{word[2:]}</span>')
                else:
                    highlighted_words.append(f'<span data-word="{word[2:]}" data-synonyms="{synonym_str}">{word[2:]}</span>')
            else:
                unchanged_sequence = []

        if show_longest_unchanged_words:
            for i in range(len(highlighted_words) - 1):
                if highlighted_words[i][6:-7] in longest_unchanged_sequence and highlighted_words[i+1][6:-7] in longest_unchanged_sequence:
                    highlighted_words[i] = f'<span class="unchanged-word">{highlighted_words[i][6:-7]}</span>'
                    highlighted_words[i+1] = f'<span class="unchanged-word">{highlighted_words[i+1][6:-7]}</span>'

        highlighted_sentence = ' '.join(highlighted_words)
        if use_yellow_highlight and any('class="changed-word"' in word for word in highlighted_words):
            highlighted_sentence = f'<span class="yellow-highlight">{highlighted_sentence}</span>'
        
        highlighted_sentences.append(highlighted_sentence)
    
    return '. '.join(highlighted_sentences)

def main():
    st.title("Kreativespace Multi-language Paraphrasing Tool - By AvinyaaEdTech")
    
    # Language selection
    languages = [
        "Auto Detect", "English", "Chinese", "Hindi", "French", "German", "Spanish", "Japanese", "Russian",
        "Italian", "Korean", "Hebrew", "Arabic", "Portuguese", "Dutch", "Finnish", "Czech", "Danish",
        "Swedish", "Polish", "Romanian", "Vietnamese", "Indonesian", "Turkish", "Greek", "Norwegian",
        "Hungarian", "Slovak", "Estonian", "Lithuanian", "Gujarati", "Bengali", "Kannada", "Tamil", "Telugu"
    ]
    input_language = st.selectbox("Select Input Language (or Auto Detect)", languages)
    output_language = st.selectbox("Select Output Language", languages[1:])  # Exclude "Auto Detect" for output
    
    # Paraphrasing mode
    modes = ["Standard", "Fluency", "Natural", "Formal", "Academic", "Simple", "Creative", "Expand", "Shorten", "Custom"]
    mode = st.selectbox("Select Paraphrasing Mode", modes)
    
    # Custom mode instructions
    custom_instructions = ""
    if mode == "Custom":
        custom_instructions = st.text_area("Enter custom paraphrasing instructions")
    
    # Expand/Shorten options
    expand_shorten_option = ""
    if mode == "Expand":
        expand_shorten_option = st.selectbox("Expand mode works best for:", ["Essays", "Research reports", "Descriptive writing", "Item descriptions", "Idea generation"])
    elif mode == "Shorten":
        expand_shorten_option = st.selectbox("Shorten mode works best for:", ["Professional presentations", "Summaries", "Marketing material"])
    
    # Synonym level
    synonym_level = st.slider("Synonym Level", 1, 5, 3)
    
    # Input text
    input_text = st.text_area("Enter text to paraphrase (max 250 words)", height=200)
    
    # Word count
    word_count = len(input_text.split())
    st.write(f"Input word count: {word_count}/{WORD_LIMIT}")

    if word_count > WORD_LIMIT:
        st.warning(f"Please limit your input to {WORD_LIMIT} words.")
        input_text = ' '.join(input_text.split()[:WORD_LIMIT])
        st.write(f"Input truncated to {WORD_LIMIT} words.")
    
    # Additional options
    col1, col2, col3 = st.columns(3)
    with col1:
        paraphrase_quotations = st.checkbox("Paraphrase quotations", value=False)
    with col2:
        avoid_contractions = st.checkbox("Avoid contractions", value=False)
    with col3:
        prefer_active_voice = st.checkbox("Prefer active voice", value=False)
    
    # Interface options
    st.subheader("Interface Options")
    use_yellow_highlight = st.checkbox("Use yellow highlight", value=True)
    show_tooltips = st.checkbox("Show tooltips", value=False)
    show_legend = st.checkbox("Show legend", value=True)
    show_changed_words = st.checkbox("Show changed words", value=True)
    show_longest_unchanged_words = st.checkbox("Show longest unchanged words", value=True)

    if 'paraphrased_text' not in st.session_state:
        st.session_state.paraphrased_text = ""
        st.session_state.highlighted_text = ""
        st.session_state.changed_words = []

    if st.button("Paraphrase"):
        if input_text and word_count <= WORD_LIMIT:
            # Detect or use selected input language
            if input_language == "Auto Detect":
                detected_lang_code = detect_language(input_text)
                detected_lang = get_full_language_name(detected_lang_code)
                st.write(f"Detected input language: {detected_lang}")
                input_language = detected_lang
            else:
                st.write(f"Input language: {input_language}")

            with st.spinner("Processing..."):
                st.session_state.paraphrased_text = paraphrase_and_translate(input_text, input_language, output_language, mode, synonym_level, expand_shorten_option, custom_instructions, paraphrase_quotations, avoid_contractions, prefer_active_voice)
            
            st.subheader(f"Paraphrased Text ({output_language}):")
            st.session_state.highlighted_text = highlight_differences_with_synonyms(input_text, st.session_state.paraphrased_text, use_yellow_highlight, show_changed_words, show_longest_unchanged_words)
            
            # Extract changed words and their synonyms
            st.session_state.changed_words = re.findall(r'<span class="changed-word tooltip" data-word="([^"]*)" data-synonyms="([^"]*)">', st.session_state.highlighted_text)

            # Add custom CSS for highlighting and tooltips
            st.markdown("""
            <style>
            .changed-word { color: green; font-weight: bold; }
            .yellow-highlight { background-color: yellow; }
            .unchanged-word { color: blue; }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(st.session_state.highlighted_text, unsafe_allow_html=True)

            if show_legend:
                st.markdown("""
                <div style="margin-top: 20px;">
                    <h4>Legend:</h4>
                    <p><span style="color: green;">●</span> Changed Words</p>
                    <p><span style="color: blue;">●</span> Longest Unchanged Words</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Word count of paraphrased text
            paraphrased_word_count = len(st.session_state.paraphrased_text.split())
            st.write(f"Output word count: {paraphrased_word_count}")
            
            # Copy button
            st.button("Copy Paraphrased Text", on_click=lambda: st.write("Text copied to clipboard!"))
        else:
            if not input_text:
                st.warning("Please enter some text to paraphrase.")
            elif word_count > WORD_LIMIT:
                st.warning(f"Please limit your input to {WORD_LIMIT} words.")

    # Synonym selection (only shown when show_tooltips is True)
    if show_tooltips and st.session_state.changed_words:
        st.subheader("Synonym Selection")
        word_to_replace = st.selectbox("Select a word to replace:", [word for word, _ in st.session_state.changed_words])
        if word_to_replace:
            synonyms = dict(st.session_state.changed_words)[word_to_replace].split(',')
            if synonyms:
                new_word = st.selectbox(f"Choose a synonym for '{word_to_replace}':", synonyms)
                if st.button("Replace Word"):
                    st.session_state.highlighted_text = re.sub(
                        f'<span class="changed-word tooltip" data-word="{word_to_replace}" data-synonyms="[^"]*">{word_to_replace}</span>',
                        f'<span class="changed-word tooltip" data-word="{new_word}" data-synonyms="{",".join(synonyms)}">{new_word}</span>',
                        st.session_state.highlighted_text
                    )
                    st.markdown(st.session_state.highlighted_text, unsafe_allow_html=True)
                    st.write(f"Replaced '{word_to_replace}' with '{new_word}'")
            else:
                st.write(f"No synonyms available for '{word_to_replace}'")

if __name__ == "__main__":
    main()