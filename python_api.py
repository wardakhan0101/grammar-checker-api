from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import language_tool_python
import json
from typing import Dict, List
import uvicorn

app = FastAPI(title="Grammar Checker API")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the grammar checker (load models once at startup)
# # [Copy all your methods from the original script here]
class SpokenEnglishGrammarChecker:
    def __init__(self):
        print("Loading language models...")
        self.nlp = spacy.load("en_core_web_sm")
        print("[DEBUG] spaCy model loaded successfully!")
        
        print("[DEBUG] ====== INITIALIZING LANGUAGETOOL ======")
        try:
            self.tool = language_tool_python.LanguageTool('en-US')
            print(f"[DEBUG] LanguageTool initialized successfully!")
            print(f"[DEBUG] LanguageTool object: {self.tool}")
            print(f"[DEBUG] LanguageTool type: {type(self.tool)}")
            
            # Test it immediately with a known error
            print("[DEBUG] Running test check: 'I has a car'")
            test_matches = self.tool.check("I has a car")
            print(f"[DEBUG] Test check found {len(test_matches)} errors")
            
            if len(test_matches) > 0:
                print(f"[DEBUG] ✓ LanguageTool is WORKING - detected errors in test")
                for match in test_matches:
                    print(f"[DEBUG]   - Rule: {match.ruleId if hasattr(match, 'ruleId') else 'N/A'}")
            else:
                print(f"[WARNING] ✗ LanguageTool found 0 errors in 'I has a car' - might not be working!")
            
        except Exception as e:
            print(f"[ERROR] LanguageTool failed to initialize: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise
        
        print("[DEBUG] ====== LANGUAGETOOL READY ======")
        print("Models loaded successfully!")
   
    def analyze_grammar(self, text: str, debug: bool = False) -> Dict:
        """
        Main function to analyze grammar mistakes in spoken English text
        Returns a detailed report with all mistakes found
        """
        # Check grammar using LanguageTool
        print(f"[DEBUG] ====== LANGUAGETOOL CHECK ======")
        print(f"[DEBUG] Input text: '{text}'")
        print(f"[DEBUG] LanguageTool object: {self.tool}")
        
        matches = self.tool.check(text)
        
        print(f"[DEBUG] LanguageTool returned {len(matches)} matches")
        for i, match in enumerate(matches):
            print(f"[DEBUG] Match {i+1}:")
            print(f"[DEBUG]   - Rule ID: {match.ruleId if hasattr(match, 'ruleId') else 'N/A'}")
            print(f"[DEBUG]   - Message: {match.message if hasattr(match, 'message') else 'N/A'}")
            print(f"[DEBUG]   - Category: {match.category if hasattr(match, 'category') else 'N/A'}")
        print(f"[DEBUG] ====== END LANGUAGETOOL CHECK ======\n")
        
        # Process with spaCy for additional context
        doc = self.nlp(text)
        
        # Debug mode - print token analysis
        if debug:
            print("\n--- DEBUG: Token Analysis ---")
            for i, token in enumerate(doc):
                print(f"{i}: '{token.text}' | POS: {token.pos_} | TAG: {token.tag_} | Lemma: {token.lemma_}")
            print("--- End Debug ---\n")
        
        # Custom rules - add our own intelligence
        custom_mistakes = self._check_custom_rules(doc, text)
        
        # Extract detailed mistake information
        mistakes = []
        
        # Add custom mistakes first
        mistakes.extend(custom_mistakes)
        
        # Add LanguageTool mistakes
        for match in matches:
            # Skip only pure style/formality issues, not actual grammar error
            if self._is_pure_style_issue(match):
                continue
    
            mistake = {
                'error_type': match.category if hasattr(match, 'category') else 'Grammar',
                'rule_id': match.ruleId if hasattr(match, 'ruleId') else 'UNKNOWN',
                'message': match.message if hasattr(match, 'message') else str(match),
                'mistake_text': text[match.offset:match.offset + match.errorLength] if hasattr(match, 'offset') and hasattr(match, 'errorLength') else '',
                'context': match.context if hasattr(match, 'context') else text,
                'position': {
                    'start': match.offset if hasattr(match, 'offset') else 0,
                    'end': match.offset + match.errorLength if hasattr(match, 'offset') and hasattr(match, 'errorLength') else 0
                },
                'suggestions': match.replacements[:3] if hasattr(match, 'replacements') else [],
                'severity': 'high' if 'grammar' in (match.category if hasattr(match, 'category') else '').lower() else 'medium'
            }
            mistakes.append(mistake)
        
        # Remove duplicate mistakes (same position and text)
        mistakes = self._remove_duplicates(mistakes)
        
        # Calculate metrics
        word_count = len([token for token in doc if not token.is_punct and not token.is_space])
        sentence_count = len(list(doc.sents))
        
        # Calculate grammar score
        if word_count > 0:
            mistake_rate = len(mistakes) / word_count
            score = max(0, min(100, 100 - (mistake_rate * 50)))  # Adjusted scoring
        else:
            score = 0
        
        # Generate corrected text
        # First apply custom corrections to the original text
        corrected_text = self._apply_custom_corrections(text, custom_mistakes)
        
        # Then apply LanguageTool corrections
        corrected_text = self.tool.correct(corrected_text)
        
        # Create comprehensive report
        report = {
            'original_text': text,
            'corrected_text': corrected_text,
            'mistakes': mistakes,
            'summary': {
                'total_mistakes': len(mistakes),
                'word_count': word_count,
                'sentence_count': sentence_count,
                'grammar_score': round(score, 1)
            },
            'mistake_categories': self._categorize_mistakes(mistakes)
        }
        
        return report
    
    def _check_custom_rules(self, doc, text: str) -> List[Dict]:
        """
        Custom grammar rules to catch mistakes LanguageTool might miss
        """
        mistakes = []
        
        # ====================================================================================
        # CHECK FOR SUBJECT-VERB AGREEMENT ERRORS (She don't -> She doesn't)
        # ====================================================================================
        
        # Third person singular subjects (he, she, it, + singular nouns) need -s/-es verbs
        # or auxiliary "does"/"doesn't" (not "do"/"don't")
        
        third_person_singular = ['he', 'she', 'it']
        
        for i, token in enumerate(doc):
            # Pattern 1: "She/He/It + don't" (should be "doesn't")
            if token.text.lower() in third_person_singular and i + 1 < len(doc):
                next_token = doc[i + 1]
                
                if next_token.text.lower() == "don't":
                    mistakes.append({
                        'error_type': 'GRAMMAR',
                        'rule_id': 'CUSTOM_SUBJECT_VERB_AGREEMENT_DONT',
                        'message': f"'{token.text.capitalize()}' requires 'doesn't', not 'don't'",
                        'mistake_text': f"{token.text} don't",
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f"{token.text} doesn't"],
                        'severity': 'high'
                    })
            
            # Pattern 2: "She/He/It + base verb" (should be verb+s/es)
            # Example: "She like" -> "She likes", "He go" -> "He goes"
            if token.text.lower() in third_person_singular and i + 1 < len(doc):
                next_token = doc[i + 1]
                
                # Check if next token is a base form verb (VB tag)
                if next_token.tag_ == 'VB' and next_token.pos_ == 'VERB':
                    # Skip modal verbs (can, will, should, etc.) and auxiliaries
                    if next_token.text.lower() not in ['be', 'can', 'could', 'will', 'would', 
                                                       'shall', 'should', 'may', 'might', 'must']:
                        # Get the third person form
                        verb_base = next_token.text.lower()
                        
                        # Simple rule for adding -s or -es
                        if verb_base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
                            verb_3rd = verb_base + 'es'
                        elif verb_base.endswith('y') and len(verb_base) > 1 and verb_base[-2] not in 'aeiou':
                            verb_3rd = verb_base[:-1] + 'ies'
                        else:
                            verb_3rd = verb_base + 's'
                        
                        # Special cases
                        irregular_3rd = {
                            'have': 'has',
                            'do': 'does',
                            'go': 'goes',
                            'be': 'is'
                        }
                        
                        if verb_base in irregular_3rd:
                            verb_3rd = irregular_3rd[verb_base]
                        
                        mistakes.append({
                            'error_type': 'GRAMMAR',
                            'rule_id': 'CUSTOM_SUBJECT_VERB_AGREEMENT',
                            'message': f"'{token.text.capitalize()}' is third person singular and requires '{verb_3rd}', not '{verb_base}'",
                            'mistake_text': f"{token.text} {next_token.text}",
                            'context': text,
                            'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                            'suggestions': [f"{token.text} {verb_3rd}"],
                            'severity': 'high'
                        })
            
            # Pattern 3: Singular noun + don't/base verb
            # Example: "The student don't like" -> "The student doesn't like"
            if token.tag_ == 'NN' and i + 1 < len(doc):  # Singular noun
                next_token = doc[i + 1]
                
                # Check for "don't" after singular noun
                if next_token.text.lower() == "don't":
                    mistakes.append({
                        'error_type': 'GRAMMAR',
                        'rule_id': 'CUSTOM_SUBJECT_VERB_AGREEMENT_DONT',
                        'message': f"Singular noun '{token.text}' requires 'doesn't', not 'don't'",
                        'mistake_text': f"{token.text} don't",
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f"{token.text} doesn't"],
                        'severity': 'high'
                    })
                
                # Check for base verb after singular noun (without auxiliary)
                elif next_token.tag_ == 'VB' and next_token.pos_ == 'VERB':
                    # Skip if there's an auxiliary before
                    has_auxiliary = i > 0 and doc[i - 1].text.lower() in ['will', 'would', 'can', 
                                                                           'could', 'should', 'may', 
                                                                           'might', 'must', 'do', 'does']
                    
                    if not has_auxiliary and next_token.text.lower() not in ['be', 'can', 'will', 'would']:
                        verb_base = next_token.text.lower()
                        
                        # Get third person form (same logic as above)
                        if verb_base.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
                            verb_3rd = verb_base + 'es'
                        elif verb_base.endswith('y') and len(verb_base) > 1 and verb_base[-2] not in 'aeiou':
                            verb_3rd = verb_base[:-1] + 'ies'
                        else:
                            verb_3rd = verb_base + 's'
                        
                        irregular_3rd = {
                            'have': 'has',
                            'do': 'does',
                            'go': 'goes',
                            'be': 'is'
                        }
                        
                        if verb_base in irregular_3rd:
                            verb_3rd = irregular_3rd[verb_base]
                        
                        mistakes.append({
                            'error_type': 'GRAMMAR',
                            'rule_id': 'CUSTOM_SUBJECT_VERB_AGREEMENT',
                            'message': f"Singular noun '{token.text}' requires '{verb_3rd}', not '{verb_base}'",
                            'mistake_text': f"{token.text} {next_token.text}",
                            'context': text,
                            'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                            'suggestions': [f"{token.text} {verb_3rd}"],
                            'severity': 'high'
                        })
        
        # ====================================================================================
        # END OF SUBJECT-VERB AGREEMENT CHECK
        # ====================================================================================
        
        # Countries that don't use 'the'
        countries_no_article = {
            'japan', 'china', 'india', 'france', 'germany', 'italy', 'spain', 
            'russia', 'brazil', 'canada', 'mexico', 'australia', 'korea', 
            'pakistan', 'england', 'scotland', 'ireland', 'portugal', 'norway',
            'sweden', 'denmark', 'finland', 'poland', 'turkey', 'egypt',
            'iran', 'iraq', 'syria', 'vietnam', 'thailand', 'malaysia',
            'singapore', 'argentina', 'chile', 'peru', 'colombia'
        }
        
        # Countries that DO use 'the'
        countries_with_article = {
            'united states', 'united kingdom', 'netherlands', 'philippines',
            'czech republic', 'dominican republic', 'maldives', 'bahamas',
            'congo', 'gambia', 'sudan', 'ukraine', 'vatican'
        }
        
        # Check for incorrect article usage with countries
        for i, token in enumerate(doc):
            if token.text.lower() == 'the' and i + 1 < len(doc):
                next_token = doc[i + 1]
                
                # Check if next word is a country without article
                if next_token.text.lower() in countries_no_article:
                    mistakes.append({
                        'error_type': 'Article Usage',
                        'rule_id': 'CUSTOM_COUNTRY_ARTICLE',
                        'message': f"Don't use 'the' before '{next_token.text}'",
                        'mistake_text': f'the {next_token.text}',
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [next_token.text],
                        'severity': 'medium'
                    })
                
                # Check for multi-word countries (e.g., "the South Korea")
                if i + 2 < len(doc):
                    two_word = f"{next_token.text.lower()} {doc[i + 2].text.lower()}"
                    if two_word in countries_no_article:
                        mistakes.append({
                            'error_type': 'Article Usage',
                            'rule_id': 'CUSTOM_COUNTRY_ARTICLE',
                            'message': f"Don't use 'the' before '{next_token.text} {doc[i + 2].text}'",
                            'mistake_text': f'the {next_token.text} {doc[i + 2].text}',
                            'context': text,
                            'position': {'start': token.idx, 'end': doc[i + 2].idx + len(doc[i + 2].text)},
                            'suggestions': [f'{next_token.text} {doc[i + 2].text}'],
                            'severity': 'medium'
                        })
        
        # Check for "very much + adjective" (should be "very + adjective")
        for i, token in enumerate(doc):
            if token.text.lower() == 'very' and i + 1 < len(doc):
                if doc[i + 1].text.lower() == 'much' and i + 2 < len(doc):
                    if doc[i + 2].pos_ == 'ADJ':
                        mistakes.append({
                            'error_type': 'Word Choice',
                            'rule_id': 'CUSTOM_VERY_MUCH_ADJ',
                            'message': "Use 'very' instead of 'very much' before adjectives",
                            'mistake_text': f'very much {doc[i + 2].text}',
                            'context': text,
                            'position': {'start': token.idx, 'end': doc[i + 2].idx + len(doc[i + 2].text)},
                            'suggestions': [f'very {doc[i + 2].text}'],
                            'severity': 'medium'
                        })
        
        # Check for "more better/more worse" (double comparative)
        for i, token in enumerate(doc):
            if token.text.lower() == 'more' and i + 1 < len(doc):
                next_word = doc[i + 1].text.lower()
                if next_word in ['better', 'worse', 'bigger', 'smaller', 'faster', 'slower']:
                    mistakes.append({
                        'error_type': 'Comparative',
                        'rule_id': 'CUSTOM_DOUBLE_COMPARATIVE',
                        'message': f"Don't use 'more' with '{next_word}' (already comparative)",
                        'mistake_text': f'more {next_word}',
                        'context': text,
                        'position': {'start': token.idx, 'end': doc[i + 1].idx + len(doc[i + 1].text)},
                        'suggestions': [next_word],
                        'severity': 'high'
                    })
        
        # Check for "didn't went/doesn't went" (double past tense)
        for i, token in enumerate(doc):
            if token.text.lower() in ["didn't", "doesn't", "don't"] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.tag_ == 'VBD':  # Past tense verb
                    base_form = next_token.lemma_
                    mistakes.append({
                        'error_type': 'Verb Form',
                        'rule_id': 'CUSTOM_DOUBLE_PAST',
                        'message': f"After '{token.text}', use base form '{base_form}' not past tense",
                        'mistake_text': f"{token.text} {next_token.text}",
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f"{token.text} {base_form}"],
                        'severity': 'high'
                    })
        
        # Check for wrong verb form after gonna/wanna/gotta
        # spaCy splits "gonna" into "gon" + "na", so check for both patterns
        for i, token in enumerate(doc):
            token_lower = token.text.lower()
            
            # Pattern 1: Single token (wanna, gotta)
            if token_lower in ['gonna', 'wanna', 'gotta'] and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.tag_ in ['VBZ', 'VBD']:
                    base_form = next_token.lemma_
                    mistakes.append({
                        'error_type': 'Verb Form',
                        'rule_id': 'CUSTOM_GONNA_VERB_FORM',
                        'message': f"After '{token.text}', use base form '{base_form}' not '{next_token.text}'",
                        'mistake_text': f"{token.text} {next_token.text}",
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f"{token.text} {base_form}"],
                        'severity': 'high'
                    })
            
            # Pattern 2: Split tokens - "gon" + "na" (gonna), "wan" + "na" (wanna)
            if token_lower in ['gon', 'wan', 'got'] and i + 1 < len(doc):
                if doc[i + 1].text.lower() == 'na' and i + 2 < len(doc):
                    next_verb = doc[i + 2]
                    if next_verb.tag_ in ['VBZ', 'VBD']:
                        base_form = next_verb.lemma_
                        informal_word = token.text + 'na'
                        mistakes.append({
                            'error_type': 'Verb Form',
                            'rule_id': 'CUSTOM_GONNA_VERB_FORM',
                            'message': f"After '{informal_word}', use base form '{base_form}' not '{next_verb.text}'",
                            'mistake_text': f"{informal_word} {next_verb.text}",
                            'context': text,
                            'position': {'start': token.idx, 'end': next_verb.idx + len(next_verb.text)},
                            'suggestions': [f"{informal_word} {base_form}"],
                            'severity': 'high'
                        })
            
            # Also check for "ta" pattern (gotta = got + ta)
            if token_lower == 'got' and i + 1 < len(doc):
                if doc[i + 1].text.lower() == 'ta' and i + 2 < len(doc):
                    next_verb = doc[i + 2]
                    if next_verb.tag_ in ['VBZ', 'VBD']:
                        base_form = next_verb.lemma_
                        mistakes.append({
                            'error_type': 'Verb Form',
                            'rule_id': 'CUSTOM_GONNA_VERB_FORM',
                            'message': f"After 'gotta', use base form '{base_form}' not '{next_verb.text}'",
                            'mistake_text': f"gotta {next_verb.text}",
                            'context': text,
                            'position': {'start': token.idx, 'end': next_verb.idx + len(next_verb.text)},
                            'suggestions': [f"gotta {base_form}"],
                            'severity': 'high'
                        })
        
        # Check for "much people" (should be "many people")
        for i, token in enumerate(doc):
            if token.text.lower() == 'much' and i + 1 < len(doc):
                next_token = doc[i + 1]
                # Check if next word is a countable plural noun
                if next_token.tag_ == 'NNS' or next_token.text.lower() in ['people', 'children', 'students', 'friends']:
                    mistakes.append({
                        'error_type': 'Quantifier',
                        'rule_id': 'CUSTOM_MUCH_MANY',
                        'message': f"Use 'many' instead of 'much' with countable nouns like '{next_token.text}'",
                        'mistake_text': f'much {next_token.text}',
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f'many {next_token.text}'],
                        'severity': 'high'
                    })
        
        # Check for "less people" (should be "fewer people")
        for i, token in enumerate(doc):
            if token.text.lower() == 'less' and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.tag_ == 'NNS' or next_token.text.lower() in ['people', 'children', 'students', 'items', 'things']:
                    mistakes.append({
                        'error_type': 'Quantifier',
                        'rule_id': 'CUSTOM_LESS_FEWER',
                        'message': f"Use 'fewer' instead of 'less' with countable nouns like '{next_token.text}'",
                        'mistake_text': f'less {next_token.text}',
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f'fewer {next_token.text}'],
                        'severity': 'medium'
                    })
        
        # Check for missing articles before singular countable nouns
        for i, token in enumerate(doc):
            # Check if it's a singular noun without article/determiner
            if token.tag_ == 'NN' and i > 0:
                prev_token = doc[i - 1]
                # Common subjects/objects that need articles
                if prev_token.pos_ not in ['DET', 'PRON'] and token.text.lower() in [
                    'student', 'teacher', 'doctor', 'engineer', 'book', 'car', 
                    'house', 'phone', 'computer', 'job', 'friend', 'problem'
                ]:
                    # Check if it's after "am/is/are/was/were"
                    if prev_token.text.lower() in ['am', 'is', 'are', 'was', 'were', 'become', 'became']:
                        mistakes.append({
                            'error_type': 'Article Missing',
                            'rule_id': 'CUSTOM_MISSING_ARTICLE',
                            'message': f"Add 'a' or 'an' before '{token.text}'",
                            'mistake_text': token.text,
                            'context': text,
                            'position': {'start': token.idx, 'end': token.idx + len(token.text)},
                            'suggestions': [f'a {token.text}'],
                            'severity': 'high'
                        })
        
        # Check for "since" with time duration (should use "for")
        for i, token in enumerate(doc):
            if token.text.lower() == 'since' and i + 1 < len(doc):
                next_token = doc[i + 1]
                # Check if directly followed by a plural time unit (Pattern 1: "since hours")
                if next_token.text.lower() in ['years', 'months', 'weeks', 'days', 'hours', 'minutes']:
                    mistakes.append({
                        'error_type': 'Preposition',
                        'rule_id': 'CUSTOM_SINCE_FOR',
                        'message': "Use 'for' with duration, 'since' with specific time point (e.g., 'since Monday', 'since 2020')",
                        'mistake_text': f'since {next_token.text}',
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f'for {next_token.text}'],
                        'severity': 'high'
                    })
        
                # Check for number/quantifier + time unit (Pattern 2: "since two hours")
                elif next_token.text.lower() in ['two', 'three', 'four', 'five', 'six', 'seven', 
                                                 'eight', 'nine', 'ten', 'many', 'several', 
                                                 'a', 'an', 'few'] and i + 2 < len(doc):
                    duration_word = doc[i + 2].text.lower()
                    if duration_word in ['years', 'months', 'weeks', 'days', 'hours', 'minutes',
                                'year', 'month', 'week', 'day', 'hour', 'minute']:
                        mistakes.append({
                        'error_type': 'Preposition',
                        'rule_id': 'CUSTOM_SINCE_FOR',
                        'message': "Use 'for' with duration, 'since' with specific time point",
                        'mistake_text': f'since {next_token.text} {duration_word}',
                        'context': text,
                        'position': {'start': token.idx, 'end': doc[i + 2].idx + len(doc[i + 2].text)},
                        'suggestions': [f'for {next_token.text} {duration_word}'],
                        'severity': 'high'
                        })
        
        # Check for "make" vs "do" common collocations
        for i, token in enumerate(doc):
            if token.lemma_.lower() == 'make' and i + 1 < len(doc):
                next_word = doc[i + 1].text.lower()
                
                # Check for possessive pronouns (my, his, her, your, their, our)
                check_word = next_word
                if next_word in ['my', 'his', 'her', 'your', 'their', 'our', 'the'] and i + 2 < len(doc):
                    check_word = doc[i + 2].text.lower()
                    possessive = next_word
                else:
                    possessive = None
                
                # Things we "do" not "make"
                if check_word in ['homework', 'exercise', 'business', 'favor', 'research', 'work']:
                    if possessive:
                        mistakes.append({
                            'error_type': 'Collocation',
                            'rule_id': 'CUSTOM_MAKE_DO',
                            'message': f"Use 'do {possessive} {check_word}' not 'make {possessive} {check_word}'",
                            'mistake_text': f'{token.text} {possessive} {check_word}',
                            'context': text,
                            'position': {'start': token.idx, 'end': doc[i + 2].idx + len(doc[i + 2].text)},
                            'suggestions': [f'do {possessive} {check_word}'],
                            'severity': 'medium'
                        })
                    else:
                        mistakes.append({
                            'error_type': 'Collocation',
                            'rule_id': 'CUSTOM_MAKE_DO',
                            'message': f"Use 'do {check_word}' not 'make {check_word}'",
                            'mistake_text': f'{token.text} {check_word}',
                            'context': text,
                            'position': {'start': token.idx, 'end': doc[i + 1].idx + len(doc[i + 1].text)},
                            'suggestions': [f'do {check_word}'],
                            'severity': 'medium'
                        })
        
        # Check for "say me" / "tell to me" (should be "tell me" / "say to me")
        for i, token in enumerate(doc):
            if token.lemma_.lower() == 'say' and i + 1 < len(doc):
                if doc[i + 1].text.lower() in ['me', 'him', 'her', 'us', 'them']:
                    mistakes.append({
                        'error_type': 'Verb Pattern',
                        'rule_id': 'CUSTOM_SAY_TELL',
                        'message': f"Use 'say to {doc[i + 1].text}' or 'tell {doc[i + 1].text}' (without 'to')",
                        'mistake_text': f'{token.text} {doc[i + 1].text}',
                        'context': text,
                        'position': {'start': token.idx, 'end': doc[i + 1].idx + len(doc[i + 1].text)},
                        'suggestions': [f'tell {doc[i + 1].text}', f'say to {doc[i + 1].text}'],
                        'severity': 'high'
                    })
        
        # Check for "explain me" (should be "explain to me")
        for i, token in enumerate(doc):
            if token.lemma_.lower() == 'explain' and i + 1 < len(doc):
                if doc[i + 1].text.lower() in ['me', 'him', 'her', 'us', 'them']:
                    mistakes.append({
                        'error_type': 'Verb Pattern',
                        'rule_id': 'CUSTOM_EXPLAIN_TO',
                        'message': "Use 'explain to me' not 'explain me'",
                        'mistake_text': f'{token.text} {doc[i + 1].text}',
                        'context': text,
                        'position': {'start': token.idx, 'end': doc[i + 1].idx + len(doc[i + 1].text)},
                        'suggestions': [f'{token.text} to {doc[i + 1].text}'],
                        'severity': 'high'
                    })
        
        # Check for "discuss about" (should be just "discuss")
        for i, token in enumerate(doc):
            if token.lemma_.lower() == 'discuss' and i + 1 < len(doc):
                if doc[i + 1].text.lower() == 'about':
                    mistakes.append({
                        'error_type': 'Preposition',
                        'rule_id': 'CUSTOM_DISCUSS_ABOUT',
                        'message': "'Discuss' doesn't need 'about' - use 'discuss something' directly",
                        'mistake_text': f'{token.text} about',
                        'context': text,
                        'position': {'start': token.idx, 'end': doc[i + 1].idx + len(doc[i + 1].text)},
                        'suggestions': [token.text],
                        'severity': 'medium'
                    })
        
        # Check for common wrong verb-preposition combinations
        wrong_prepositions = {
            'marry': {'wrong': ['with'], 'correct': 'to', 'note': 'or no preposition'},
            'good': {'wrong': ['in'], 'correct': 'at', 'note': ''},
            'interested': {'wrong': ['about', 'on', 'for'], 'correct': 'in', 'note': ''},
            'different': {'wrong': ['than', 'with'], 'correct': 'from', 'note': ''},
            'angry': {'wrong': ['on'], 'correct': 'with/at', 'note': ''},
            'arrive': {'wrong': ['to'], 'correct': 'at/in', 'note': ''},
            'listen': {'wrong': [''], 'correct': 'to', 'note': ''},
            'wait': {'wrong': [''], 'correct': 'for', 'note': ''},
        }
        
        for i, token in enumerate(doc):
            lemma = token.lemma_.lower()
            if lemma in wrong_prepositions and i + 1 < len(doc):
                next_token = doc[i + 1]
                if next_token.pos_ == 'ADP':  # It's a preposition
                    prep = next_token.text.lower()
                    rule = wrong_prepositions[lemma]
                    
                    if prep in rule['wrong']:
                        correct = rule['correct']
                        note = f" {rule['note']}" if rule['note'] else ""
                        mistakes.append({
                            'error_type': 'Preposition',
                            'rule_id': 'CUSTOM_VERB_PREP',
                            'message': f"Use '{lemma} {correct}'{note}, not '{lemma} {prep}'",
                            'mistake_text': f'{token.text} {prep}',
                            'context': text,
                            'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                            'suggestions': [f'{token.text} {correct}'],
                            'severity': 'high'
                        })
        
        # Check for "informations" (uncountable)
        for token in doc:
            if token.text.lower() == 'informations':
                mistakes.append({
                    'error_type': 'Uncountable Noun',
                    'rule_id': 'CUSTOM_UNCOUNTABLE',
                    'message': "'Information' is uncountable - no 's' at the end",
                    'mistake_text': token.text,
                    'context': text,
                    'position': {'start': token.idx, 'end': token.idx + len(token.text)},
                    'suggestions': ['information'],
                    'severity': 'high'
                })
            elif token.text.lower() in ['advices', 'furnitures', 'equipments', 'homeworks']:
                base_form = token.text[:-1]  # Remove 's'
                mistakes.append({
                    'error_type': 'Uncountable Noun',
                    'rule_id': 'CUSTOM_UNCOUNTABLE',
                    'message': f"'{base_form}' is uncountable - don't add 's'",
                    'mistake_text': token.text,
                    'context': text,
                    'position': {'start': token.idx, 'end': token.idx + len(token.text)},
                    'suggestions': [base_form],
                    'severity': 'high'
                })
        
        # Check for incorrect word order in indirect questions
        # Pattern: "know/tell/ask/wonder + WH-word + verb + subject" (wrong)
        # Should be: "know/tell/ask/wonder + WH-word + subject + verb"
        reporting_verbs = ['know', 'tell', 'ask', 'wonder', 'understand', 'remember', 'forget', 'think', 'guess', 'imagine']
        question_words = ['where', 'when', 'why', 'how', 'what', 'who', 'which']
        
        for i, token in enumerate(doc):
            if token.lemma_.lower() in reporting_verbs and i + 1 < len(doc):
                # Look for question word after the verb
                for j in range(i + 1, min(i + 3, len(doc))):
                    if doc[j].text.lower() in question_words and j + 2 < len(doc):
                        # Check if next token is a verb (auxiliary or main)
                        next_token = doc[j + 1]
                        following_token = doc[j + 2]
                        
                        # Pattern: WH-word + AUX/VERB + SUBJECT (wrong order)
                        if next_token.pos_ in ['AUX', 'VERB'] and following_token.pos_ in ['DET', 'PRON', 'NOUN', 'PROPN']:
                            # This is likely wrong word order
                            wh_word = doc[j].text
                            verb = next_token.text
                            subject_start = following_token.text
                            
                            mistakes.append({
                                'error_type': 'Word Order',
                                'rule_id': 'CUSTOM_INDIRECT_QUESTION',
                                'message': f"In indirect questions, use '{wh_word} + subject + verb', not '{wh_word} + verb + subject'",
                                'mistake_text': f'{wh_word} {verb} {subject_start}',
                                'context': text,
                                'position': {'start': doc[j].idx, 'end': following_token.idx + len(following_token.text)},
                                'suggestions': [f'{wh_word} {subject_start} {verb}'],
                                'severity': 'high'
                            })
                        break
        
        # Check for "other" without article before singular countable noun
        # Should be "another" or "the other"
        for i, token in enumerate(doc):
            if token.text.lower() == 'other' and i + 1 < len(doc):
                next_token = doc[i + 1]
                # Check if followed by singular countable noun
                if next_token.tag_ == 'NN' and next_token.pos_ == 'NOUN':
                    # Check if there's no article before "other"
                    has_article = i > 0 and doc[i - 1].text.lower() in ['the', 'an', 'a']
                    if not has_article:
                        mistakes.append({
                            'error_type': 'Article/Determiner',
                            'rule_id': 'CUSTOM_OTHER_ANOTHER',
                            'message': f"Use 'another {next_token.text}' or 'the other {next_token.text}', not 'other {next_token.text}'",
                            'mistake_text': f'other {next_token.text}',
                            'context': text,
                            'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                            'suggestions': [f'another {next_token.text}', f'the other {next_token.text}'],
                            'severity': 'high'
                        })
        
        # Check for "each" or "every" with plural nouns
        # Should be singular
        for i, token in enumerate(doc):
            if token.text.lower() in ['each', 'every'] and i + 1 < len(doc):
                next_token = doc[i + 1]
                # Check if followed by plural noun
                if next_token.tag_ == 'NNS':  # Plural noun
                    singular = next_token.lemma_
                    mistakes.append({
                        'error_type': 'Singular/Plural',
                        'rule_id': 'CUSTOM_EACH_EVERY_SINGULAR',
                        'message': f"'{token.text}' is used with singular nouns, not plural",
                        'mistake_text': f'{token.text} {next_token.text}',
                        'context': text,
                        'position': {'start': token.idx, 'end': next_token.idx + len(next_token.text)},
                        'suggestions': [f'{token.text} {singular}'],
                        'severity': 'high'
                    })
        
        # Check for double negatives
        # Pattern: negative verb (don't, doesn't, didn't, etc.) + negative word (nothing, nobody, never, etc.)
        negative_words = ['nothing', 'nobody', 'nowhere', 'never', 'neither', 'none', 'no one']
        
        for i, token in enumerate(doc):
            # Check if it's a negative auxiliary/verb
            if token.text.lower() in ["don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "haven't", "hasn't", "hadn't"]:
                # Look ahead for negative words
                for j in range(i + 1, min(i + 8, len(doc))):
                    if doc[j].text.lower() in negative_words:
                        # Found double negative
                        positive_form = {
                            'nothing': 'anything',
                            'nobody': 'anybody',
                            'nowhere': 'anywhere',
                            'never': 'ever',
                            'no one': 'anyone'
                        }.get(doc[j].text.lower(), 'anything')
                        
                        mistakes.append({
                            'error_type': 'Double Negative',
                            'rule_id': 'CUSTOM_DOUBLE_NEGATIVE',
                            'message': f"Avoid double negatives. Use '{positive_form}' instead of '{doc[j].text}' with negative verbs",
                            'mistake_text': f"{token.text} ... {doc[j].text}",
                            'context': text,
                            'position': {'start': token.idx, 'end': doc[j].idx + len(doc[j].text)},
                            'suggestions': [positive_form],
                            'severity': 'high'
                        })
                        break
        
        # Check for incorrect adjective order
        # Standard order: Opinion > Size > Age > Shape > Color > Origin > Material > Purpose
        adjective_categories = {
            # Size adjectives
            'size': ['big', 'small', 'large', 'tiny', 'huge', 'little', 'tall', 'short', 'long', 'wide', 'narrow'],
            # Age adjectives
            'age': ['old', 'new', 'young', 'ancient', 'modern', 'recent'],
            # Color adjectives
            'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink', 'purple', 'orange', 'grey', 'gray'],
            # Shape adjectives
            'shape': ['round', 'square', 'circular', 'rectangular', 'triangular', 'oval'],
            # Origin adjectives
            'origin': ['american', 'chinese', 'japanese', 'indian', 'french', 'german', 'british', 'english', 'italian', 'spanish'],
            # Material adjectives
            'material': ['wooden', 'metal', 'plastic', 'glass', 'cotton', 'silk', 'leather', 'paper', 'stone', 'steel', 'gold', 'silver']
        }
        
        # Order priority (lower number = comes first)
        order_priority = {
            'opinion': 0,
            'size': 1,
            'age': 2,
            'shape': 3,
            'color': 4,
            'origin': 5,
            'material': 6,
            'purpose': 7
        }
        
        def get_adj_category(adj_text):
            adj_lower = adj_text.lower()
            for category, words in adjective_categories.items():
                if adj_lower in words:
                    return category
            # If not found in specific categories, assume opinion
            return 'opinion'
        
        # Find sequences of adjectives before nouns
        for i, token in enumerate(doc):
            if token.pos_ == 'ADJ' and i + 1 < len(doc):
                # Collect consecutive adjectives
                adj_sequence = [token]
                j = i + 1
                while j < len(doc) and doc[j].pos_ == 'ADJ':
                    adj_sequence.append(doc[j])
                    j += 1
                
                # Check if there are at least 2 adjectives
                if len(adj_sequence) >= 2:
                    # Get categories for each adjective
                    adj_info = [(adj, get_adj_category(adj.text)) for adj in adj_sequence]
                    
                    # Check if order is correct
                    for k in range(len(adj_info) - 1):
                        curr_adj, curr_cat = adj_info[k]
                        next_adj, next_cat = adj_info[k + 1]
                        
                        curr_priority = order_priority.get(curr_cat, 0)
                        next_priority = order_priority.get(next_cat, 0)
                        
                        # If current should come after next, it's wrong order
                        if curr_priority > next_priority:
                            # Found incorrect order
                            wrong_order = ' '.join([adj.text for adj, _ in adj_info])
                            correct_order = ' '.join([adj.text for adj, _ in sorted(adj_info, key=lambda x: order_priority.get(x[1], 0))])
                            
                            mistakes.append({
                                'error_type': 'Adjective Order',
                                'rule_id': 'CUSTOM_ADJ_ORDER',
                                'message': f"Adjective order: {curr_cat} adjectives usually come after {next_cat} adjectives",
                                'mistake_text': wrong_order,
                                'context': text,
                                'position': {'start': adj_sequence[0].idx, 'end': adj_sequence[-1].idx + len(adj_sequence[-1].text)},
                                'suggestions': [correct_order],
                                'severity': 'medium'
                            })
                            break  # Only report once per sequence
        
        return mistakes
    
    def _apply_custom_corrections(self, text: str, custom_mistakes: List[Dict]) -> str:
        """
        Apply corrections for custom mistakes to generate corrected text
        """
        corrected = text
        
        # Sort mistakes by position (reverse order to avoid offset issues)
        sorted_mistakes = sorted(custom_mistakes, key=lambda x: x['position']['start'], reverse=True)
        
        for mistake in sorted_mistakes:
            if mistake['suggestions']:
                start = mistake['position']['start']
                end = mistake['position']['end']
                suggestion = mistake['suggestions'][0]  # Use first suggestion
                
                # Replace the mistake with the suggestion
                corrected = corrected[:start] + suggestion + corrected[end:]
        
        return corrected
    
    def _remove_duplicates(self, mistakes: List[Dict]) -> List[Dict]:
        """
        Remove duplicate mistakes that have the same position and mistake text
        Keep the one with highest severity or most detailed message
        """
        if not mistakes:
            return mistakes
        
        unique_mistakes = {}
        
        for mistake in mistakes:
            # Create a key based on overlapping positions, not exact match
            start = mistake['position']['start']
            end = mistake['position']['end']
            mistake_text = mistake['mistake_text'].lower().strip()
            
            # Check if this mistake overlaps with any existing ones
            found_overlap = False
            for existing_key in list(unique_mistakes.keys()):
                existing_start, existing_end, existing_text = existing_key
                
                # Check if positions overlap and text is similar
                if (start <= existing_end and end >= existing_start) and \
                   (mistake_text in existing_text or existing_text in mistake_text):
                    found_overlap = True
                    # Keep the one with higher severity or longer text (more context)
                    existing = unique_mistakes[existing_key]
                    if mistake['severity'] == 'high' and existing['severity'] != 'high':
                        del unique_mistakes[existing_key]
                        unique_mistakes[(start, end, mistake_text)] = mistake
                    elif len(mistake['mistake_text']) > len(existing['mistake_text']):
                        # Keep the more detailed one
                        del unique_mistakes[existing_key]
                        unique_mistakes[(start, end, mistake_text)] = mistake
                    break
            
            if not found_overlap:
                unique_mistakes[(start, end, mistake_text)] = mistake
        
        return list(unique_mistakes.values())
    
    def _is_pure_style_issue(self, match) -> bool:
        """
        Only filter out pure style/formality suggestions, not grammar errors
        """
        category = match.category.upper()
        
        # Only ignore these specific categories
        if category in ['STYLE', 'TYPOGRAPHY', 'REDUNDANCY']:
            return True
        
        return False
    
    def _categorize_mistakes(self, mistakes: List[Dict]) -> Dict:
        """
        Group mistakes by category for better reporting
        """
        categories = {}
        for mistake in mistakes:
            category = mistake['error_type']
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
        return categories
    
    def generate_text_report(self, analysis: Dict) -> str:
        """
        Generate a human-readable text report
        """
        report = []
        report.append("=" * 70)
        report.append("SPOKEN ENGLISH GRAMMAR ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Summary section
        summary = analysis['summary']
        report.append("SUMMARY:")
        report.append(f"  Grammar Score: {summary['grammar_score']}/100")
        report.append(f"  Total Mistakes: {summary['total_mistakes']}")
        report.append(f"  Words: {summary['word_count']} | Sentences: {summary['sentence_count']}")
        report.append("")
        
        # Original vs Corrected
        report.append("ORIGINAL TEXT:")
        report.append(f"  {analysis['original_text']}")
        report.append("")
        report.append("CORRECTED TEXT:")
        report.append(f"  {analysis['corrected_text']}")
        report.append("")
        
        # Detailed mistakes
        if analysis['mistakes']:
            report.append("DETAILED MISTAKES:")
            report.append("-" * 70)
            
            for i, mistake in enumerate(analysis['mistakes'], 1):
                report.append(f"\n{i}. ERROR: {mistake['mistake_text']}")
                report.append(f"   Type: {mistake['error_type']}")
                report.append(f"   Issue: {mistake['message']}")
                
                if mistake['suggestions']:
                    report.append(f"   Suggestions: {', '.join(mistake['suggestions'])}")
                
                report.append(f"   Context: ...{mistake['context']}...")
                report.append(f"   Severity: {mistake['severity']}")
        else:
            report.append("✓ No grammar mistakes found! Excellent work!")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def get_json_report(self, analysis: Dict) -> str:
        """
        Return analysis as formatted JSON (for API/Flutter integration later)
        """
        return json.dumps(analysis, indent=2)

    
    # Add all your helper methods here (_check_custom_rules, _apply_custom_corrections, etc.)
    # For brevity, I'm not copying them all, but you should include them


# Initialize checker once at startup
checker = SpokenEnglishGrammarChecker()

# Request/Response models
class TextRequest(BaseModel):
    text: str
    debug: bool = False

class GrammarResponse(BaseModel):
    original_text: str
    corrected_text: str
    mistakes: List[Dict]
    summary: Dict
    mistake_categories: Dict

@app.get("/")
async def root():
    return {
        "message": "Grammar Checker API",
        "status": "online",
        "endpoints": {
            "/analyze": "POST - Analyze grammar",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    """
    Health check endpoint - also tests if LanguageTool is working
    """
    try:
        # Test LanguageTool
        test_text = "I has a car"
        test_matches = checker.tool.check(test_text)
        
        languagetool_status = {
            "working": len(test_matches) > 0,
            "test_text": test_text,
            "errors_found": len(test_matches),
            "rules_triggered": [match.ruleId for match in test_matches] if hasattr(test_matches[0] if test_matches else None, 'ruleId') else []
        }
        
        return {
            "status": "healthy", 
            "models_loaded": True,
            "spacy_loaded": checker.nlp is not None,
            "languagetool_status": languagetool_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "models_loaded": False
        }

@app.post("/analyze", response_model=GrammarResponse)
async def analyze_text(request: TextRequest):
    """
    Analyze text for grammar mistakes
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Analyze the grammar
        result = checker.analyze_grammar(request.text, debug=request.debug)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/quick-check")
async def quick_check(request: TextRequest):
    """
    Quick grammar check - returns just corrected text and error count
    Useful for real-time corrections in chat
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = checker.analyze_grammar(request.text)
        
        return {
            "corrected_text": result['corrected_text'],
            "has_errors": len(result['mistakes']) > 0,
            "error_count": len(result['mistakes']),
            "score": result['summary']['grammar_score']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Check error: {str(e)}")


if __name__ == "__main__":
    # Run the API
    uvicorn.run(app, host="0.0.0.0", port=8000)