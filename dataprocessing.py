import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from CONFIG import*
# =============================================================================
# Hybrid Character Vocabulary
# =============================================================================

class HybridCharacterVocabulary:
    def __init__(self, fixed_alphabet=None, max_vocab_size=500):
        # Default alphabet if none provided
        self.default_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        # Initialize with special tokens
        self.char_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_char = {0: self.pad_token, 1: self.unk_token}
        self.n_chars = 2  # Count of special tokens

        # Character frequency tracking
        self.char_count = {}

        # Add fixed alphabet first if provided
        if fixed_alphabet is not None:
            self.add_fixed_alphabet(fixed_alphabet)
    
    def add_fixed_alphabet(self, alphabet):
        """Add a fixed alphabet to the vocabulary."""
        print(f"Adding fixed alphabet with {len(alphabet)} characters")
        
        # Add each character from the alphabet to the vocabulary
        for char in alphabet:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.char_count[char] = float('inf')  # Mark as fixed alphabet character
                self.n_chars += 1

        print(f"After adding fixed alphabet: {self.n_chars} characters")
    
    def build_from_texts(self, texts, min_count=2):
        """Build additional vocabulary from texts."""
        print("Extending vocabulary from training data...")
        
        # Count characters
        for text in texts:
            for char in text:
                if char not in self.char_count:
                    self.char_count[char] = 0
                self.char_count[char] += 1

        # Add frequently occurring characters that aren't already in the vocabulary
        chars_added = 0
        for char, count in sorted(self.char_count.items(), key=lambda x: x[1], reverse=True):
            # Skip if already in vocabulary
            if char in self.char_to_idx:
                continue
                
            # Skip if below minimum count
            if count < min_count:
                continue
                
            # Skip if we've reached maximum vocabulary size
            if self.n_chars >= self.max_vocab_size:
                break
                
            # Add to vocabulary
            self.char_to_idx[char] = self.n_chars
            self.idx_to_char[self.n_chars] = char
            self.n_chars += 1
            chars_added += 1
        
        # Add special attention to important characters for toxicity detection
        special_toxicity_chars = [
            # Common substitutions in toxic text
            '@', '0', '1', '3', '4', '$']

        for char in special_toxicity_chars:
            if char not in self.char_to_idx and self.n_chars < self.max_vocab_size:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.n_chars += 1
                chars_added += 1
            
        print(f"Added {chars_added} new characters from training data")
        print(f"Final vocabulary size: {self.n_chars} characters")
        
        # Print some statistics about character coverage
        total_chars = sum(self.char_count.values())
        covered_chars = sum(count for char, count in self.char_count.items() if char in self.char_to_idx)
        coverage = covered_chars / total_chars * 100
        print(f"Character coverage: {coverage:.2f}% of all character occurrences")

    def encode_text(self, text, max_len=300):
        """Convert text to sequence of character indices."""
        # Pre-allocate array with pad tokens
        indices = np.full(max_len, self.char_to_idx[self.pad_token], dtype=np.int64)

        # Fill with actual character indices
        for i, char in enumerate(text[:max_len]):
            indices[i] = self.char_to_idx.get(char, self.char_to_idx[self.unk_token])

        return indices

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

# =============================================================================
# Text Preprocessing
# =============================================================================

# Pre-compiled patterns for efficiency
WHITESPACE_PATTERN = re.compile(r'\s+')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
def preprocess_text(text, max_len=300):
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with special token
    text = URL_PATTERN.sub(" <URL> ", text)
       
    # Remove excessive whitespace
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    
    # Truncate if needed
    if len(text) > max_len:
        text = text[:max_len]
    
    return text

# Optional: Language detection
def detect_language(text):
    # Common Tagalog words
    tagalog_markers = [
    # Standard function words / particles and common pronouns
    "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila", 
    "ang", "ng", "sa", "mga", "ni", "namin", "natin", "nila",
    "hindi", "oo", "opo", "wala", "meron", "dahil", "kasi",
    
    # Additional function words, particles, and connectors
    "na", "nang", "lang", "lamang", "ba", "daw", "raw", "pala",
    "kaya", "pero", "ngunit", "subalit", "at", "o", "kung", 
    "kapag", "pag", "sapagkat", "para", "pwede", "puwede", 
    "baka", "siguro", "marahil", "naman", "nga", "kay", "kina", "nina",
    
    # Pronouns, demonstratives, and interrogatives
    "ito", "iyan", "iyon", 
    "sino", "ano", "saan", "kailan", "bakit", "paano", "ilan",
    
    # Common verbal affixes (use with caution for token matching)
    "mag", "nag", "um", "in", "an", "ma", "ipag", "ipa", "pa",
    
    # Additional words and expressions
    "ayaw", "paki", "salamat", "walang", "anuman", 
    "pasensya", "pasensiya", "mahal", "murang", 
    "malaki", "maliit", "masaya", "malungkot", 
    "maganda", "gwapo",
    ]
    
    # Clean and tokenize the text
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Skip very short texts
    if len(words) < 3:
        return 'en'  # Default to English
    
    # Count Tagalog markers
    tagalog_count = sum(1 for word in words if word in tagalog_markers)
    tagalog_ratio = tagalog_count / len(words)
    
    # Determine language based on ratio
    if tagalog_ratio > 0.15:  # If more than 15% words are Tagalog markers
        return 'tl'
    else:
        return 'en'

def extract_toxicity_features(text):
    """Extract additional features for toxicity detection."""
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
        
    features = {}
    
    # 1. Check for ALL CAPS segments
    words = text.split()
    all_caps_words = [w for w in words if len(w) > 2 and w.isupper()]
    features['all_caps_ratio'] = len(all_caps_words) / max(1, len(words))
    
    # 2. Toxic keyword checking
    # Define a basic toxic keyword list (expand this with your domain knowledge)
    toxic_keywords = [
    'kink', 'kinkster', 'kinky', 'kinkyJesus', 'kiss ass', 'kiss my ass', 'kissass', 'kitty puncher', 'kiunt', 'kkk',
    'kkk', 'klan', 'klan', 'klansman', 'klansmen', 'klanswoman', 'klanswomen', 'klitoris', 'klootzak', 'kneegrows',
    'knickers', 'knob', 'knob eater', 'knob end', 'knob gobbler', 'knob jockey', 'knob-face', 'knob-gobbler', 'knob-head', 'knob3d',
    'knob3nd', 'knobbing', 'knobd', 'knobe', 'knobead', 'knobead', 'knobeads', 'knobed', 'knobed', 'knobeds',
    'knobend', 'knobend', 'knobend', 'knobender', 'knobends', 'knobendy', 'knobendz', 'knober', 'knobes', 'knobgobbler',
    'knobhead', 'knobhead', 'knobheads', 'knobjockies', 'knobjocky', 'knobjocky', 'knobjokey', 'knobjokey', 'knobjokeys',
    'knobz', 'knockers', 'knulle', 'kock', 'kondum', 'kondums', 'kooch', 'kooches', 'koon', 'kootch',
    'krap', 'krappy', 'kraut', 'krauts', 'ku kluxer', 'kuffar', 'kuk', 'kuksuger', 'kum', 'kumbubble',
    'kumbullbe', 'kumer', 'kummer', 'kumming', 'kums', 'kunilingus', 'kunnilingus', 'kunt', 'kunts', 'kuntz',
    'kupal', 'kurac', 'kurwa', 'kushi', 'kushis', 'kusi', 'kwa', 'kwai lo', 'kwai los', 'kwif',
    'kyke', 'kyke', 'kykes', 'kyopo', 'kyopos', 'kyrpa', 'l3i + ch', 'l3i+ch', 'l3i+ch', 'l3i\\+ch',
    'l3itch', 'l3itch', 'l3itches', 'l@dyb0i', 'l@dyb0y', 'l@dyboy', 'labia', 'labia', 'ladboys', 'ladboyz',
    'ladiboy', 'lady-boy', 'ladyb0i', 'ladyb0y', 'ladyboy', 'ladyboys', 'ladyboyz', 'lapdance', 'leather restraint', 'leather straight',
    'leatherrestraint', 'lebos', 'lech', 'leche', 'leching', 'lechugas', 'lemon party', 'lemonparty', 'leper', 'lesbain',
    'lesbayn', 'lesbin', 'lesbo', 'lesbo', 'lesbos', 'lez', 'lezbe', 'lezbefriends', 'lezbian', 'lezbians',
    'lezbo', 'lezbos', 'lezz', 'lezzian', 'lezzie', 'lezzies', 'lezzo', 'lezzy', 'libido', 'licker',
    'licking', 'lickme', 'lilniglet', 'limey', 'limpdick', 'limy', 'lingerie', 'lintik', 'lipshits', 'lipshitz',
    'livesex', 'lmao', 'lmfao', 'loadedgun', 'lolita', 'loose woman', 'lovebone', 'lovegoo', 'lovegun', 'lovejuice',
    'lovemuscle', 'lovepistol', 'loverocket', 'lowlife', 'lsd', 'lubejob', 'lubra', 'lucifer', 'luckycammeltoe', 'lugan',
    'lugans', 'lusting', 'lusty', 'lynch', 'm-fucking', 'm0f0', 'm0f0', 'm0f0s', 'm0fo', 'm0fo',
    'm0foes', 'm0fos', 'm0ng0l0id', 'm0ngoloid', 'm0thafucked', 'm0thafucker', 'm0thafucking', 'm0therfuckeds', 'm0therfucker', 'm0therfucking',
    'm0therfvcker', 'm45terbate', 'm@asterbated', 'm@derfaker', 'm@derfuck', 'm@derfuckers', 'ma5terb8', 'ma5terbate', 'mabuno', 'mabunos',
    'macaca', 'macacas', 'mafugly', 'magicwand', 'mahbuno', 'mahbunos', 'make me come', 'makemecome', 'makemecum', 'male squirting',
    'mamhoon', 'mams', 'man chowder', 'man meat', 'man seed', 'manhater', 'manpaste', 'maricon', 'maricón', 'marijuana',
    'markasses', 'masochist', 'masokist', 'massa', 'massterbait', 'masstrbait', 'masstrbate', 'mastabate', 'mastabater', 'master-bate',
    'masterb8', 'masterbaiter', 'masterbat', 'masterbat3', 'masterbate', 'masterbates', 'masterbating', 'masterbation', 'masterbations', 'masterblaster',
    'mastrabator', 'masturbat', 'masturbate', 'masturbating', 'masturbation', 'mattressprincess', 'mau mau', 'mau maus', 'maumau', 'maumaus',
    'mcfagget', 'meat curtains', 'meat-sword', 'meatbeatter', 'meatrack', 'mecha fag', 'mega fag', 'menage', 'merd', "mf'er",
    "mf'ers", "mf'ing", 'mfckers', 'mfing', 'mfk', 'mfs', 'mfukk', 'mfukker', 'mgger', 'mggor',
    'mibun', 'mick', 'mickeyfinn', 'mideast', 'mierda', 'milf', 'milf', 'mindfuck', 'mindfuck', 'minge',
    'minger', 'mo-fo', 'mockey', 'mockie', 'mocky', 'mof0', 'mof0es', 'mof0s', 'mofcker', 'mofo',
    'mofo', 'mofo ass', 'mofoes', 'mofos', 'mofoshit', 'mofuccers', 'mofucckers', 'mofuck', 'mofucker', 'mofuckkas',
    'mofuk', 'mofukkas', 'moky', 'molest', 'molestation', 'molester', 'molester', 'molestor', 'moneyshot', 'mong',
    'mong', 'mongoloid', 'mongrel', 'monkleigh', 'moolie', 'moon cricket', 'moon crickets', 'mooncricket', 'mooncrickets', 'moron',
    'moskal', 'moskals', 'moslem', 'mosshead', 'motha fucka', 'motha fucker', 'motha fucker', 'motha fuckers', 'motha fuker', 'motha fukkah',
    'motha fukker', 'mothaf@cked', 'mothafcked', 'mothafcking', 'mothafucced', 'mothafuccer', 'mothafuccing', 'mothafuck', 'mothafuck', 'mothafucka',
    'mothafucka', 'mothafuckas', 'mothafuckas', 'mothafuckasses', 'mothafuckaz', 'mothafuckaz', 'mothafuckazzes', 'mothafucked', 'mothafucked', 'mothafuckeds',
    'mothafucker', 'mothafucker', 'mothafuckers', 'mothafuckers', 'mothafuckin', 'mothafuckin', 'mothafucking', 'mothafucking', 'mothafuckings', 'mothafuckings',
    'mothafuckins', 'mothafucks', 'mothafucks', 'mothafuckz', 'mothafvcked', 'mother effer', 'mother fuck', 'mother fuck you', 'mother fucka', 'mother fucker',
    'mother fucker', 'mother fuckers', 'mother fucking', 'mother fukah', 'mother fuker', 'mother fukkah', 'mother fukker', 'mother-fucker', 'mothercker', 'motherf@kka',
    'motherfacking', 'motherfcked', 'motherfckin', 'motherfcking', 'motherfcks', 'motherfckshit', 'motherfecka', 'motherfecker', 'motherfk', 'motherfucca',
    'motherfuccas', 'motherfuccers', 'motherfuck', 'motherfuck', 'motherfucka', 'motherfucked', 'motherfucked', 'motherfuckeds', 'motherfucker', 'motherfucker',
    'motherfuckers', 'motherfuckers', 'motherfuckin', 'motherfuckin', 'motherfucking', 'motherfucking', 'motherfuckings', 'motherfuckings', 'motherfuckingshit', 'motherfuckins',
    'motherfuckka', 'motherfuckka', 'motherfuckkas', 'motherfuckkers', 'motherfucks', 'motherfucks', 'motherfukka', 'motherfukker', 'motherfukkings', 'motherfvck',
    'motherfvcked', 'motherfvckeds', 'motherfvcker', 'motherfvcker', 'motherfvckers', 'motherfvcking', 'motherfxck', 'motherfxcking', 'motherlovebone', 'mothfck',
    'mothrfucker', 'mothter fuck', 'mouliewop', 'mound of venus', 'moundofvenus', 'mr hands', 'mrhands', 'mtherfucker', 'mtherfuker', 'mthrfcker',
    'mthrfuck', 'mthrfucker', 'mthrfucking', 'mtrfck', 'mtrfuck', 'mtrfucker', 'muddafukkas', 'mudderfuk', 'mudderfukker', 'mufdive',
    'mufdivin', 'muff', 'muff', 'muff', 'muff diver', 'muffdive', 'muffdiver', 'muffdiving', 'muffdiving', 'muffdivings',
    'muffindiver', 'muffindivin', 'muffindiving', 'mufflikcer', 'muffpuff', 'muhfucking', 'muie', 'mulatto', 'mulkku', 'muncher',
    'munging', 'munt', 'munter', 'muschi', 'mushroom tip', 'mutha fucka', 'mutha fucker', 'mutha fucker', 'mutha fukah', 'mutha fuker',
    'mutha fukkah', 'mutha fukker', 'muthafecker', 'muthafecker', 'muthafeckers', 'muthafucka', 'muthafuckaz', 'muthafucker', 'muthafuckers', 'muthafuckings',
    'muthafuckker', 'muthafuckker', 'muthafuckkers', 'muthafukka', 'muther', 'mutherfucker', 'mutherfucker', 'mutherfuckers', 'mutherfucking', 'muthrfucking',
    'mzungu', 'mzungus', 'n0bhead', 'n0bj0cky', 'n1ckker', 'n1g3r', 'n1g3rz', 'n1gg3r', 'n1gg3rs', "n1gg@",
    "n1gg@hs", 'n1gga', 'n1gga', 'n1ggah', 'n1ggahs', 'n1ggas', 'n1ggazes', 'n1gger', 'n1gger', 'n1ggers',
    'n1gguh', 'n1gr', 'n3gro', 'nads', 'nakakaburat', 'naked', 'nambla', 'nastt', 'nastybitch', 'nastyho',
    'nastyslut', 'nastywhore', 'nawashi', 'nazi', 'nazis', 'nazism', 'necro', 'needthedick', 'negga', 'neggar',
    'negr0', 'negres', 'negress', 'negro', 'negro', 'negroes', 'negroes', 'negroid', 'negroid', 'negros',
    'neonazi', 'nepesaurio', 'niccer', 'nicka', 'nickas', 'nicker', 'nickk3r', 'nickker', 'nig', 'nig',
    'nig nog', 'nig-nog', 'niga', 'niga', 'nigah', 'nigar', 'nigars', 'nigas', 'nigasses', 'nigers',
    'nigers', 'nigette', 'nigettes', 'nigg', 'nigg3r', 'nigg3r', 'nigg3rs', 'nigg4h', 'nigg4h', 'nigg4hs',
    'nigg@', 'nigg@hs', 'nigg@s', 'nigg@z', 'nigg@zzes', 'nigga', 'nigga', 'nigga', 'nigga lover', 'niggah',
    'niggah', 'niggahs', 'niggahs', 'niggahz', 'niggar', 'niggaracci', 'niggard', 'niggarded', 'niggarding', 'niggardliness',
    'niggardlinesss', 'niggardly', 'niggards', 'niggars', 'niggas', 'niggas', 'niggass', 'niggaz', 'niggaz', 'niggazzes',
    'nigger', 'nigger', 'nigger', 'nigger lover', 'niggerhead', 'niggerhole', 'niggers', 'niggers', 'niggerz', 'niggir',
    'niggle', 'niggled', 'niggles', 'nigglings', 'niggor', 'niggress', 'niggress', 'niggresses', 'nigguh', 'nigguh',
    'nigguhs', 'nigguhs', 'nigguhz', 'niggur', 'niggurs', 'niglet', 'niglet', 'nignigs', 'nignog', 'nignog',
    'nigor', 'nigors', 'nigr', 'nigra', 'nigra', 'nigras', 'nigre', 'nigre', 'nigres', 'nigress',
    'nigs', 'nigs', 'niguh', 'nigur', 'niiger', 'niigr', 'nikk3r', 'nikkas', 'nikker', 'nimal',
    'nimphomania', 'nimrod', 'ninny', 'nip', 'nipple', 'nipplering', 'nipples', 'nips', 'nittit', 'nlgger',
    'nlggor', 'nob', 'nob jockey', 'nob jokey', 'nob jokeys', 'nobbyhead', 'nobhead', 'nobhead', 'nobheads', 'nobj0key',
    'nobjockies', 'nobjocky', 'nobjocky', 'nobjokey', 'nobjokey', 'nobjokeys', 'nobs', 'nofuckingway', 'nog', 'nonce',
    'nookey', 'nookie', 'nooky', 'noonan', 'nooner', 'nsfw', 'nsfw images', 'nuckas', 'nudger', 'nudie',
    'nudies', 'nuggets', 'numbnuts', 'nut butter', 'nut sack', 'nutbutter', 'nutfucker', 'nutsack', 'nutsack', 'nutsacks',
    'nutten', 'nympho', 'nympho', 'nymphomania', 'nymphomaniac', 'o c k', 'octopussy', 'octopussy', 'ogag', 'olok',
    'omg', 'omorashi', 'one cup two girls', 'one guy', 'one guy one jar', 'one jar', 'ontherag', 'orafis', 'orga', 'orgasim',
    'orgasim;', 'orgasims', 'orgasm', 'orgasmic', 'orgasms', 'orgasum', 'orgies', 'orgy', 'oriface', 'orifiss', 'orospu',
    'osama', 'oven dodger', 'ovum', 'ovums', 'p e n i s', 'p i s', 'p u s s y', 'p.u.s.s.y.', 'p0rn',
    'p3n1shead', 'p3nisfcker', 'p3nisfcukers', 'p3nisfvcker', 'p3nisfvckers', 'pack my fudge', 'packerfudgehead', 'packi', 'packie', 'packing fudge',
    'packing fudgehead', 'packingfudge', 'packingfudgefucker', 'packingfudgefucking', 'packsomefudgefucker', 'packy', 'paddy', 'paedophile', 'paki', 'paki',
    'pakie', 'pakingshet', 'pakis', 'pakshet', 'paky', 'pakyu', 'palesimian', 'palm jockey', 'pancake face', 'pancake face',
    'pancake faces', 'panooch', 'pansies', 'pansy', 'panti', 'pantie', 'panties', 'panty', 'paska', 'payo',
    'pcp', 'pearlnecklace', 'pecker', 'pecker', 'peckerhead', 'peckerhead', 'peckerwood', 'pedo', 'pedo', 'pedobear', 'pedobear',
    'pedobears', 'pedophile', 'pedophilia', 'pedophiliac', 'pedophl', 'pedos', 'pedoz', 'peeenus', 'peeenusss', 'peehole',
    'peen', 'peener', 'peenus', 'peepee', 'peepshow', 'peepshpw', 'pegging', 'peinus', 'pen1s', 'penas',
    'pendejo', 'pendy', 'penetrate', 'penetration', 'peni5', 'penial', 'penile', 'penis', 'penis', 'penis-breath',
    'penises', 'penisfcker', 'penisfuccer', 'penisfucker', 'penisfucker', 'penisfuckers', 'penisfvcker', 'penisfvckers', 'penishead', 'penisland',
    'penislick', 'penislicker', 'penispuffer', 'penthouse', 'penus', 'penuus', 'perse', 'perv', 'perversion', 'pesteng yawa',
    'peter', 'peter puffer', 'peyote', 'ph@ggots', 'phaggot', 'phaggots', 'phagot', 'phags', 'phalli', 'phallic',
    'phone sex', 'phonesex', 'phuc', 'phuc', 'phucc', 'phuccer', 'phucchead', 'phuccing', 'phuck', 'phuck',
    'phuck3r', 'phucked', 'phucker', 'phuckin', 'phucking', 'phuckings', 'phucks', 'phucup', 'phuk', 'phuk',
    'phuked', 'phuked', 'phukeds', 'phuker', 'phukhead', 'phuking', 'phuking', 'phukings', 'phukk', 'phukked', 'phukked',
    'phukkeds', 'phukker', 'phukker', 'phukking', 'phuks', 'phuks', 'phukshit', 'phuku', 'phukup', 'phungky',
    'phuq', 'phuq', 'phuqs', 'phvckings', 'pi55', 'picaninny', 'piccaninny', 'picka', 'pickaninnies', 'pickaninny',
    'piece of shit', 'pieceofshit', 'piefke', 'piefkes', 'pierdol', 'pig fucker', 'pigfucker', 'pigfucker', 'pigfuckers',
    'pigfucking', 'pigfukker', 'piggyfuck', 'pigshit', 'piker', 'pikey', 'piky', 'pillow biter', 'pillow-biter', 'pillowbiter',
    'pillu', 'pimmel', 'pimp', 'pimped', 'pimper', 'pimpis', 'pimpjuic', 'pimpjuice', 'pimpsimp', 'pindick',
    'pinko', 'pis', 'pises', 'pisin', 'pising', 'pisof', 'piss', 'piss', 'piss face', 'piss off fuckhead',
    'piss pig', 'piss shit', 'piss-off', 'pissed', 'pisser', 'pissers', 'pisses', 'pissflap', 'pissflaps', 'pisshead',
    'pissin', 'pissing', 'pissoff', 'pissoff', 'pissoffs', 'pisspig', 'pizda', 'playboy', 'playgirl', 'pleasure chest',
    'pleasurechest', 'pocha', 'pochas', 'pocho', 'pochos', 'pocketpool', 'pohm', 'pohms', 'poke', 'poki', 'pokpok',
    'polac', 'polack', 'polacks', 'polak', 'pole licker', 'pole smoker', 'pole smoker', 'pole sucker', 'polesmoker', 'polesmoker',
    'pollock', 'pollocks', 'pommie grant', 'pommie grants', 'pommy', 'ponyplay', 'poof', 'poon', 'poonani', 'poonany',
    'poontang', 'poontsee', 'poop', 'poop chute', 'poopchute', 'pooper', 'pooperscooper', 'pooping', 'poorwhitetrash', 'popimp',
    'porch monkey', 'porch monkey', 'porch monkies', 'porchmonkey', 'porn', 'pornflick', 'pornking', 'porno', 'pornography', 'pornos',
    'pornprincess', 'pound town', 'poundtown', 'poyet', 'pplicker', 'pr0n', 'pr1c', 'pr1ck', 'pr1k', 'prairie nigger',
    'prairie niggers', 'preteen', 'pric', 'prick', 'prick', 'prick-face', 'prick-gobbler', 'prick-head', 'prickhead', 'pricks',
    'pricks', 'prig', 'prince albert piercing', 'pron', 'prostitute', 'pthc', 'pu$sy', "pu'keng", 'pu55i', 'pu55y',
    'pu55y', 'pube', 'pube', 'pubes', 'pubic', 'pubiclice', 'pubis', 'pucha', 'puchanggala', 'puchangina',
    'pudboy', 'pudd', 'puddboy', 'puke', 'puki', 'pukinangina', 'puking', 'pula', 'pull the pud', 'punani',
    'punani', 'punanny', 'punany', 'punk ass mofoes', 'punkass', 'punkasses', 'punky', 'punta', 'puntang', 'punyeta',
    'purinapricness', 'pusies', 'puss', 'puss', 'pusse', 'pussee', 'pusses', 'pussi', 'pussie', 'pussie', 'pussies',
    'pussies', 'pussless', 'pusslicker', 'pussy', 'pussy', 'pussy', 'pussy cat', 'pussy fucker', 'pussy lick', 'pussy licker',
    'pussy licking', 'pussycat', 'pussydestroyer', 'pussyeater', 'pussyfart', 'pussyfuck', 'pussyfucker', 'pussylicker', 'pussylickers', 'pussylicking',
    'pussylips', 'pussylover', 'pussypalace', 'pussypounder', 'pussys', 'pussys', 'pussywhipped', 'pusy', 'puta', 'puta',
    'puta', 'putang', 'putang ina', 'putangina', 'putanginamo', 'putaragis', 'puto', 'putragis', 'puuke', 'puuker',
    'puussy', 'puyet', 'puzzies', 'puzzy', 'qahbeh', 'quashie', 'queaf', 'queef', 'queer', 'queerasses',
    'queerhole', 'queero', 'queers', 'queers', 'queerz', 'quickie', 'quicky', 'quiff', 'quim', 'qweers',
    'qweerz', 'qweir', 'r-tard', 'r-tards', 'r3t@rd', 'r3t@rded', 'r3tard', 'r5e', 'ra8s', 'raghead',
    'raghead', 'ragheads', 'ragheads', 'ragtard', 'ramrod', 'rape', 'raped', 'raper', 'raping', 'rapist',
    'rat bastard', 'rat baztad', 'ratbu', 'rautenberg', 'reacharound', 'rearend', 'rearentry', 'recktum', 'rectal', 'rectum',
    'rectum', 'rectus', 'redleg', 'redlegs', 'redlight', 'redskin', 'redskin', 'redskins', 'reefer', 'reestie',
    'reetard', 'reich', 'renob', 'rentafuck', 'rere', 'retard', 'retard', 'retarded', 'retardo', 'retardotron',
    'retards', 'retardz', 'reverse cowgirl', 'reversecowgirl', 'rice monkey', 'rim job', 'rimjaw', 'rimjob', 'rimming', 'ritard',
    'rosebuds', 'rosy palm', 'rosy palm and her 5 sisters', 'rosypalm', 'rosypalmandher5sisters', 'rosypalmandherefivesisters', 'round eyes', 'roundeye', 'rtard', 'rtards',
    'rumprammer', 'ruski', 'russki', 'russkie', 'rusty trombone', 'rustytrombone', 's h i t', 's hit', 's hit', 's&m',
    's-h-1-t', 's-h-i-t', 's-lut', 's-o-b', 's.h.i.t.', 's.o.b.', 's.o.b.', 's.o.b.s', 's/h/i/t', 's0b',
    's_h_i_', 's_h_i_s', 's_h_i_t', 'sack', 'sadis', 'sadism', 'sadist', 'sadom', 'salad tosser', 'sambo',
    'sambo', 'sambos', 'samckdaddy', 'sanchez', 'sand nigger', 'sand nigger', 'sand niggers', 'sandm', 'sandnigger', 'santorum',
    'sausage jockey', 'sausagequeen', 'scag', 'scallywag', 'scamfuck', 'scank', 'scantily', 'scat', 'schaffer', 'scheiss',
    'schizo', 'schlampe', 'schlong', 'schlong', 'schmuck', 'schvartse', 'schvartsen', 'schwartze', 'schwartzen', 'scissoring',
    'screwyou', 'scroat', 'scrog', 'scrote', 'scrotum', 'scrotum', 'scrud', 'scumfuck', 'scumfucker', 'scumfvck',
    'scummy', 'scut', 'seduce', 'semen', 'seppo', 'seppos', 'septics', 'sex', 'sex', 'sexcam', 'sexed',
    'sexfarm', 'sexhound', 'sexhouse', 'sexi', 'sexing', 'sexkitten', 'sexo', 'sexpot', 'sexslave', 'sextogo',
    'sextoy', 'sextoys', 'sexual', 'sexually', 'sexwhore', 'sexx', 'sexxi', 'sexxx', 'sexxxi', 'sexxxy',
    'sexxy', 'sexy', 'sexymoma', 'sexyslim', 'sh! +', 'sh!+', 'sh!+', 'sh!t', 'sh1s', 'sh1t',
    'sh1t', 'sh1t', 'sh1t3', 'sh1td1ck', 'sh1tdick', 'sh1te', 'sh1ter', 'sh1tfuck', 'sh1th3ad',
    'sh1theads', 'sh1ts', 'sh1ts', 'sh1tsome', 'sh1tt', 'sh1tter', 'sh1tty', 'sh1tz', 'sh3mal3', 'sh3male',
    'shag', 'shagger', 'shaggin', 'shagging', 'shamedame', 'sharmuta', 'sharmute', 'shat', 'shat', 'shav',
    'shaved beaver', 'shaved pussy', 'shavedbeaver', 'shavedpussy', 'shawtypimp', 'she-male', 'sheeeet', 'sheeney', 'sheet', 'sheister',
    'shemal3', 'shemale', 'shemale', 'shemales', 'shet', 'shhit', 'shi+', 'shi+', 'shi+e', 'shi+y',
    'shiat', 'shibari', 'shibary', 'shiddick', 'shiester', 'shiesterfuck', 'shiesterfuckface', 'shiesterfuckhead', 'shiesterfucks', 'shinola',
    'shipal', 'shipdit', 'shit', 'shit', 'shit', 'shit ass', 'shit face', 'shit for brains', 'shit fuck', 'shit fucker',
    'shit head', 'shit licker', 'shit stain', 'shit-arse', 'shit-ass', 'shit-ass', 'shit-bag', 'shit-bagger', 'shit-bandit', 'shit-brain',
    'shit-breath', 'shit-cunt', 'shit-dick', 'shit-eating', 'shit-face', 'shit-faced', 'shit-fit', 'shit-fucker', 'shit-head', 'shit-heel',
    'shit-hole', 'shit-house', 'shit-load', 'shit-pot', 'shit-spitter', 'shit-stain', 'shit-stuffers', 'shit3', 'shitass', 'shitass',
    'shitasses', 'shitassfucker', 'shitassfuckface', 'shitbag', 'shitbag', 'shitbagger', 'shitbird', 'shitblimp', 'shitblimp', 'shitblimps',
    'shitbrain', 'shitbrain', 'shitbreath', 'shitcan', 'shitcunt', 'shitd1ck', 'shitdick', 'shitdick', 'shitdicks', 'shitdikk',
    'shitdip', 'shite', 'shite', 'shiteater', 'shiteating', 'shiteblimps', 'shited', 'shited', 'shitedick', 'shitefuck',
    'shitefulls', 'shitehead', 'shites', 'shitey', 'shitey', 'shitface', 'shitface', 'shitfaced', 'shitfaced', 'shitfacefuck',
    'shitfacefucker', 'shitfck', 'shitfit', 'shitfk', 'shitforbrains', 'shitfreak', 'shitfuck', 'shitfuck', 'shitfucker', 'shitfucker',
    'shitfuckhead', 'shitfuckmotherfucker', 'shitfucks', 'shitfudgefucker', 'shitfull', 'shitfvck', 'shithapens', 'shithappens', 'shithead', 'shithead',
    'shitheadfucker', 'shitheadfuckface', 'shitheads', 'shitheel', 'shithole', 'shithole', 'shithouse', 'shiting', 'shitings', 'shitlist',
    'shitload', 'shitola', 'shitoutofluck', 'shitpot', 'shits', 'shits', 'shitsdick', 'shitsfuck', 'shitsful', 'shitspitter',
    'shitstain', 'shitt', 'shittastic', 'shittasticfuck', 'shitte', 'shitted', 'shitted', 'shitter', 'shitter', 'shitterfucker',
    'shitters', 'shitti', 'shitties', 'shittiest', 'shittiest', 'shitting', 'shitting', 'shittings', 'shittings', 'shitty', 'shitty',
    'shitty mofoes', 'shittydick', 'shittydicks', 'shittyfuck', 'shittyfuckface', 'shittyful', 'shity', 'shitz', 'shiz', 'shiznit', 'shlong',
    'shmale', 'shortfuck', 'shota', 'shtfuk', 'shunga', 'shut the fuck up', 'shylock', 'shylock', 'shylocks', 'shyt', 'shyte',
    'shytfeisterfuck', 'shytty', 'shyty', 'simp', 'sira ulo', 'siraulo', 'sissy', 'sissy', 'sixsixsix', 'sixty-nine',
    'sixtynine', 'sixtyniner', 'sk@nks', 'sk@nky', 'sk@nkz', 'skag', 'skanck', 'skank', 'skank', 'skankbitch',
    'skankee', 'skankey', 'skankfuck', 'skanks', 'skanks', 'skankwhore', 'skanky', 'skanky', 'skankybitch', 'skankywhore',
    'skankz', 'skeet', 'skinflute', 'skribz', 'skullfuck', 'skum', 'skumbag', 'skurwysyn', 'skwa', 'skwe',
    'sl@nteye', 'slag', 'slag', 'slantard', 'slanteye', 'slanteye', 'slanteye b1tch', 'slanteyes', 'slanteyeshit', 'slantfreak',
    'slanty', 'slanty', 'slapper', 'sleezeball', 'slideitin', 'slimeball', 'slimebucket', 'slit', 'slopehead', 'slopeheads',
    'sloper', 'slopers', 'slopey', 'slopeys', 'slopies', 'slopy', 'slut', 'slut', 'slut', 'slut hole',
    'slutbag', 'slutbucket', 'slutdumper', 'slutkiss', 'sluts', 'sluts', 'slutt', 'slutting', 'slutty', 'slutty',
    'slutwear', 'slutwhore', 'slutz', 'smackthemonkey', 'smeg', 'smegma', 'smegma', 'smut', 'smutty', 'snatch',
    'snatch licker', 'snatchpatch', 'sniggered', 'sniggering', 'sniggers', 'snowback', 'snowballing', 'snownigger', 'snuff', 'soab',
    'socksucker', 'sodom', 'sodomise', 'sodomite', 'sodomize', 'sodomy', 'son o bitch', 'son of a bitch', 'son of a bitch', 'son of a whore',
    'son-of-a-bitch', 'son-of-a-bitch', 'son-of-a-whore', 'sonna bitch', 'sonofabitch', 'sonofabitch', 'sonofbitch', 'sonofabitches', 'sons of b1tches', 'sons of bitches',
    'sons-of-bitches', 'sonz of bitchez', 'sooties', 'soppy bollucks', 'souse', 'soused', 'soyboy', 'spac', 'spaghettibender', 'spaghettinigger',
    'spank', 'spanking', 'spankthemonkey', 'spastic', 'spearchucker', 'spearchuckers', 'sperm', 'sperm', 'spermacide', 'spermbag',
    'spermhearder', 'spermherder', 'sphencter', 'sphincter', 'spic', 'spic', 'spicfuck', 'spick', 'spick', 'spicks',
    'spics', 'spics', 'spicshit', 'spierdalaj', 'spig', 'spig', 'spigotty', 'spik', 'spik', 'spiks',
    'spix', 'splittail', 'splooge', 'spludge', 'spooge', 'spook', 'spooks', 'spread legs', 'spreadeagle', 'spunk', 'spunk',
    'spunk', 'spunky', 'sqeh', 'squa', 'squarehead', 'squareheads', 'squaw', 'squinty', 'squirting', 'stagg',
    'stfu', 'stfu', 'stiffy', 'stoned', 'stoner', 'strap on', 'strapon', 'strappado', 'strip club', 'stripclub',
    'stroking', 'stuinties', 'stump chewer', 'stupid fucker', 'stupid hoe', 'stupidasses', 'stupidfuck', 'stupidfucker', 'style doggy', 'suck',
    'suck my cock', 'suck my d', 'suck my dick', 'suck off', 'suckdick', 'sucked', 'sucker', 'sucking', 'suckme', 'suckmyass',
    'suckmydick', 'suckmytit', 'suckoff', 'suicide girl', 'suicide girls', 'suicidegirl', 'suicidegirls', 'suka', 'sultrywoman',
    'sultrywomen', 'sum of a bitch', 'sumbitch', 'sumofabiatch', 'suso', 'susu', 'swallower', 'swalow', 'swamp guinea', 'swamp guineas',
    'swastika', 'swine', 'swine fucker', 'syphilis', 't i t', 't i ts', 't1t', 't1tt1e5', 't1tties', 'tacohead',
    'tacohead', 'tacoheads', 'tadger', 'tae', 'taena', 'taff', 'take off your', 'taking the piss', 'tallywacker', 'tamod',
    'tanga', 'tangina', 'tar babies', 'tar baby', 'tar-baby', 'taragis', 'tarantado', 'tarbaby', 'tard', 'tard',
    'tard asses', 'tart', 'taste my', 'tastemy', 'tawdry', 'tea bagging', 'teabagging', 'teat', 'teets', 'teez',
    'terd', 'teste', 'testee', 'testes', 'testical', 'testicle', 'testicles', 'testis', 'tete', 'teti',
    'text', 'thicklip', 'thicklips', 'thirdeye', 'thirdleg', 'threesome', 'threeway', 'throat yogurt', 'throater', 'throating',
    'thumbzilla', 'thundercunt', 'tickle the pickle', 'tied up', 'tig ol bitties', 'tig old bitties', 'tight white', 'timang', 'timber nigger', 'timber nigger',
    'timber niggers', 'timbernigger', 'tinil', 'tit', 'tit', 'titbitnipply', 'tite', 'titfuck', 'titfucker', 'titfuckin',
    'titi', 'titi', 'titjob', 'titlicker', 'titlover', 'tits', 'tits', 'titt', 'tittie', 'tittie5',
    'tittiefucker', 'titties', 'tittis', 'titty', 'tittyfuck', 'tittyfucker', 'tittys', 'tittywank', 'titwank', 'tity',
    'to murder', 'tongethruster', 'tongue fucker', 'tongue fucking', 'tongue in a', 'tongueina', 'tonguethrust', 'tonguetramp', 'toots', 'topless',
    'tortur', 'torture', 'tosser', 'tosser', 'tosser', 'tossing salad', 'towel head', 'towel heads', 'towelhead', 'towelhead',
    'towelheads', 'towelshithead', 'trailertrash', 'tramp', 'trannie', 'tranny', 'transsexual', 'transvestite', 'transvestite', 'trashb1tch',
    'trashbitch', 'trashbitches', 'trashbitchez', 'trashbtch', 'trasherbitch', 'trasherbitchs', 'trashybitches', 'tribadism', 'trisexual', 'trois',
    'trots', 'trouser snake', 'trousersnake', 'tub girl', 'tubgirl', 'tuckahoe', 'tungaw', 'tunneloflove', 'turd', 'turd burgler',
    'turdcutter', 'turdhead', 'turnon', 'tush', 'tushy', 'tw4t', 'tw@t', 'twa+', 'twat', 'twat',
    'twat', 'twat', 'twat waffle', 'twatface', 'twathead', 'twatlips', 'twats', 'twats', 'twatt', 'twattish',
    'twatty', 'twatwaffle', 'twatzilla', 'twink', 'twink', 'twinkie', 'two girls one cup', 'twobitwhore', 'twunt', 'twunter',
    'udge packer', 'ukrop', 'ulol', 'ulul', 'unclefucker', 'unfuckable', 'ungas', 'upskirt', 'upskirts', 'uptheass',
    'upthebutt', 'urethra play', 'urethraplay', 'urophilia', 'usama', 'useless fucker', 'ussys', 'uzi', 'v a g i n a', 'v14gra',
    'v1gra', 'v4gra', 'va-j-j', 'va1jina', 'vag', 'vag', 'vag1na', 'vagiina', 'vagina', 'vaj1na',
    'vajayjay', 'vajina', 'valium', 'venus mound', 'vgra', 'vibr', 'vibrater', 'vibrator', 'vigra', 'violet wand',
    'virginbreaker', 'vittu', 'vixen', 'vjayjay', 'vorarephilia', 'voyeurweb', 'voyuer', 'vullva', 'vulva', 'vulva',
    'w00se', 'w0p', 'w4nk3r', 'w4nker', 'w@nker', 'w@nkers', 'wab', 'wang', 'wang', 'wang wrangler',
    'wank', 'wank', 'wank', 'wank off', 'wank3r', 'wank3rs', 'wankbastard', 'wanked', 'wanker', 'wanker',
    'wankers', 'wankies', 'wanking', 'wanking', 'wanks', 'wanky', 'waysted', 'wazoo', 'we1back', 'weenie',
    'weenie', 'weewee', 'weiner', 'weiner', 'welcher', 'wench', 'wet back', 'wet dream', 'wetb', 'wetback',
    'wetback', 'wetbacks', 'wetbacks', 'wetdream', 'wetspot', 'wh00r', 'wh0r3', 'wh0re', 'wh0re', 'wh0reface',
    'whack off', 'whacker', 'whash', 'what the fuck', 'whigger', 'whiggers', 'whiskeydick', 'whiskydick', 'whit', 'white power',
    'white trash', 'whitenigger', 'whitepower', 'whitetrash', 'whitey', 'whiteys', 'whities', 'whoar', 'whoar', 'whoars',
    'whop', 'whor3', 'whoralicious', 'whore', 'whore', 'whore', 'whorealicious', 'whorebag', 'whored', 'whoreface',
    'whorefucker', 'whorehopper', 'whorehouse', 'whores', 'whores', 'whoring', 'wichser', 'wigga', 'wiggas', 'wigger',
    'wigger', 'wiggers', 'willie', 'willies', 'williewanker', 'willy', 'willy-whacker', 'window licker', 'wise ass', 'wnker',
    'wog', 'wogs', 'woose', 'wop', 'wop', 'wophead', 'worldsex', 'wrapping men', 'wrinkled starfish', 'wtf',
    'wtf', 'wuss', 'wuzzie', 'x-rated', 'x-rated2g1c', 'xkwe', 'xrated', 'xtc', 'xx', 'xxx',
    'xxxxxx', 'yank', 'yaoi', 'yarpie', 'yarpies', 'yed', 'yellow showers', 'yellowman', 'yellowshowers', 'yid',
    'yids', 'yiffy', 'yobbo', 'yourboobs', 'yourpenis', 'yourtits', 'yury', 'zabourah', 'zigabo',
    'zigabos', 'zip in the wire', 'zipperhead', 'zipperhead', 'zipperheads', 'zoophile', 'zoophilia', '🖕', 'tanga', 'tanginamo',
    'fukcer', '8080', 'obob', 'OBOB', 'putangina', 'kys', 'nigga', 'nigger', 'naega', 'mamatay',
    'papatayin', 'peenoise', 'fap', 'jerk', 'jerk off', 'Putang ina', 'Gago', 'Gaga', 'Tangina mo', 'Leche',
    'Pakshet', 'Tarantado', 'Ulol', 'Bobo', 'Boba', 'Lintik ka', 'Bwisit', 'Punyeta', 'Hudas', 'Lintik',
    'Sira ulo', 'Yawa', 'Tanga', 'Hampaslupa', 'Demonyo', 'Inutil', 'Shet', 'Kup*l', 'Amp*t', 'Ul*l',
    'Peste', 'Bwesit', 'Kagaguhan', 'Peste', 'Sumpain ka', 'Anak ng teteng', 'Anak ng tinapa', 'Hayop ka', 'Walang kwenta',
    'Ingrato', 'Suwapang', 'Madamot', 'Duwag', 'Salbahe', 'Burat', 'Tite mo', 'Kupal', 'Engot', 'Gunggong',
    'Abnoy', 'Timang', 'Dumbass', 'Dipshit', 'Jackass', 'Son of a bitch', 'Asshat', 'Douchebag', 'Prick', 'Scumbag',
    'Tool', 'Wanker', 'Nitwit', 'Half-wit', 'Moron', 'Blockhead', 'Knucklehead', 'Bonehead', 'Loser', 'Jerk',
    'Screw you', 'Eat shit', 'Piss off', 'Bugger off', 'Dunce', 'Idiot', 'Fool', 'Numbskull', "a$shole", 'a$$hole',
    'a$$', 'depota', 'buguk', 'bugok', 'tanaydana', 'dana', 'b0b0', 'engeng', 'eng-eng'
]

    
    # Count toxic keywords (case insensitive)
    lower_text = text.lower()
    keyword_count = 0
    detected_keywords = []
    
    for keyword in toxic_keywords:
        if keyword in lower_text:
            keyword_count += 1
            detected_keywords.append(keyword)
    
    features['toxic_keyword_count'] = keyword_count
    features['toxic_keyword_ratio'] = keyword_count / max(1, len(words))
    features['detected_keywords'] = detected_keywords
    
    return features

# =============================================================================
# Dataset and DataLoader
# =============================================================================

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels=None, char_vocab=None, max_len=300, detect_lang=False):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.detect_lang = detect_lang
        
        # Pre-process all texts
        self.processed_texts = [preprocess_text(text, max_len) for text in texts]
        
        # Extract toxicity features for all texts
        print("Extracting toxicity features...")
        self.toxicity_features = [extract_toxicity_features(text) for text in self.processed_texts]
        
        # Initialize character vocabulary if not provided
        if char_vocab is None:
            if CONFIG.get('use_hybrid_vocabulary', True):
                self.char_vocab = HybridCharacterVocabulary(
                    fixed_alphabet=CONFIG.get('alphabet', None),
                    max_vocab_size=CONFIG.get('max_vocab_size', 500)
                )
                self.char_vocab.build_from_texts(self.processed_texts, min_count=2)
            else:
                self.char_vocab = HybridCharacterVocabulary()
                self.char_vocab.build_vocab(self.processed_texts)
        else:
            self.char_vocab = char_vocab
        
        # Detect languages if enabled
        if self.detect_lang:
            print("Detecting languages for texts...")
            self.languages = [detect_language(text) for text in self.processed_texts]
            lang_counts = Counter(self.languages)
            print(f"Language distribution: {dict(lang_counts)}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Convert idx to int if it's a string
        if isinstance(idx, str):
            try:
                idx = int(idx)
            except ValueError:
                raise TypeError(f"Cannot convert idx '{idx}' to integer")
            
        # Get pre-processed text
        processed_text = self.processed_texts[idx]
            
        # Get toxicity features
        features = self.toxicity_features[idx]
            
        # Encode text to character indices
        char_ids = self.char_vocab.encode_text(processed_text, self.max_len)
            
        if self.labels is not None:
            item = {
                'char_ids': torch.tensor(char_ids, dtype=torch.long),
                'labels': torch.tensor(self.labels[idx], dtype=torch.float),
                'text': processed_text,
                'all_caps_ratio': torch.tensor(features['all_caps_ratio'], dtype=torch.float),
                'toxic_keyword_count': torch.tensor(features['toxic_keyword_count'], dtype=torch.float),
                'toxic_keyword_ratio': torch.tensor(features['toxic_keyword_ratio'], dtype=torch.float)
            }
            # Add language info if available
            if self.detect_lang:
                item['language'] = self.languages[idx]
            return item
        else:
            item = {
                'char_ids': torch.tensor(char_ids, dtype=torch.long),
                'text': processed_text,
                'all_caps_ratio': torch.tensor(features['all_caps_ratio'], dtype=torch.float),
                'toxic_keyword_count': torch.tensor(features['toxic_keyword_count'], dtype=torch.float),
                'toxic_keyword_ratio': torch.tensor(features['toxic_keyword_ratio'], dtype=torch.float)
            }
            # Add language info if available
            if self.detect_lang:
                item['language'] = self.languages[idx]
            return item

# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data_from_csv(file_path, text_column=None, toxicity_column=None, category_columns=None):
    print(f"Loading data from {file_path}...")
    
    # Get column names from CONFIG if not specified
    if text_column is None:
        text_column = CONFIG.get('text_column', 'text')
    if toxicity_column is None:
        toxicity_column = CONFIG.get('toxicity_column', 'toxicity_level')
    if category_columns is None:
        category_columns = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
    
    try:
        # Try with different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying latin-1...")
            df = pd.read_csv(file_path, encoding='latin-1')
        
        print(f"Loaded {len(df)} rows from {file_path}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise
    
    # Check if columns exist
    missing_columns = []
    if text_column not in df.columns:
        missing_columns.append(text_column)
    if toxicity_column not in df.columns:
        missing_columns.append(toxicity_column)
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Extract texts
    texts = df[text_column].tolist()
    
    # Create labels array [toxicity, insult, profanity, threat, identity_hate]
    toxicity_levels = df[toxicity_column].astype(int).values
    
    # Initialize the labels array
    labels = np.zeros((len(df), 1 + len(category_columns)))
    labels[:, 0] = toxicity_levels
    
    # Add category values if available
    for i, col in enumerate(category_columns):
        if col in df.columns:
            labels[:, i+1] = df[col].astype(int).values
        else:
            print(f"Warning: Category column '{col}' not found. Using all zeros.")
    
    # Print data distribution
    print(f"\nToxicity level distribution:")
    for level, count in sorted(Counter(toxicity_levels).items()):
        percentage = count / len(toxicity_levels) * 100
        print(f"  Level {level}: {count} examples ({percentage:.1f}%)")
    
    print(f"\nCategory distribution:")
    for i, col in enumerate(category_columns):
        positive_count = np.sum(labels[:, i+1] == 1)
        percentage = positive_count / len(labels) * 100
        print(f"  {col}: {positive_count} positive examples ({percentage:.1f}%)")
    
    return texts, labels

def create_data_loaders(texts, labels, char_vocab=None, test_size=0.2, val_size=0.25, 
                        batch_size=32, num_workers=4, max_len=300, detect_lang=False, seed=42):

    # Perform train/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels[:, 0]
    )
    
    # Perform train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=val_size, random_state=seed, stratify=train_labels[:, 0]
    )
    
    print(f"Split data into {len(train_texts)} training, {len(val_texts)} validation, "
          f"and {len(test_texts)} test examples")
    
    # Create datasets
    train_dataset = ToxicityDataset(
        train_texts, train_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    # If char_vocab wasn't provided, use the one created by the train_dataset
    if char_vocab is None:
        char_vocab = train_dataset.char_vocab
    
    val_dataset = ToxicityDataset(
        val_texts, val_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    test_dataset = ToxicityDataset(
        test_texts, test_labels, char_vocab, max_len=max_len, detect_lang=detect_lang
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, char_vocab

def create_ood_test_set(input_path, output_path, criteria='long_texts', sample_size=500):
    """
    Create an Out-Of-Distribution (OOD) test set by selecting examples based on specified criteria.
    
    Args:
        input_path (str): Path to the original data file
        output_path (str): Path where the OOD test set will be saved
        criteria (str): Criterion for selecting OOD examples:
                        'long_texts' - select texts longer than average
                        'short_texts' - select texts shorter than average
                        'rare_words' - select texts with rare vocabulary
                        'random' - randomly sample examples
        sample_size (int): Number of examples to include in the OOD test set
    """
    print(f"Creating OOD test set with criteria: {criteria}")
    
    # Load original data
    df = pd.read_csv(input_path)
    text_column = CONFIG.get('text_column', 'comment')
    
    # Calculate text lengths
    df['text_length'] = df[text_column].apply(len)
    mean_length = df['text_length'].mean()
    std_length = df['text_length'].std()
    
    # Select examples based on criteria
    if criteria == 'long_texts':
        # Select texts that are longer than average (more than 1 std above mean)
        threshold = mean_length + std_length
        filtered_df = df[df['text_length'] > threshold]
        print(f"Selected {len(filtered_df)} texts longer than {threshold:.1f} characters")
        
    elif criteria == 'short_texts':
        # Select texts that are shorter than average (more than 1 std below mean)
        threshold = max(10, mean_length - std_length)  # Ensure minimum length
        filtered_df = df[df['text_length'] < threshold]
        print(f"Selected {len(filtered_df)} texts shorter than {threshold:.1f} characters")
        
    elif criteria == 'rare_words':
        # Get vocab frequency from the entire dataset
        import re
        from collections import Counter
        
        # Tokenize and count all words
        all_words = []
        for text in df[text_column]:
            words = re.findall(r'\b\w+\b', str(text).lower())
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        # Calculate rarity score for each text (average inverse frequency of words)
        rarity_scores = []
        for text in df[text_column]:
            words = re.findall(r'\b\w+\b', str(text).lower())
            if not words:
                rarity_scores.append(0)
                continue
            
            # Calculate average rarity
            rarity = sum(1 / (word_counts[word] + 1) for word in words) / len(words)
            rarity_scores.append(rarity)
        
        df['rarity_score'] = rarity_scores
        
        # Select texts with high rarity scores
        threshold = np.percentile(rarity_scores, 75)  # Top 25% rarest
        filtered_df = df[df['rarity_score'] > threshold]
        print(f"Selected {len(filtered_df)} texts with rare vocabulary")
        
    else:  # Default to random sampling
        # Randomly sample examples
        filtered_df = df.sample(min(sample_size, len(df)), random_state=CONFIG.get('seed', 42))
        print(f"Randomly selected {len(filtered_df)} texts")
    
    # Ensure we don't have too many examples
    if len(filtered_df) > sample_size:
        filtered_df = filtered_df.sample(sample_size, random_state=CONFIG.get('seed', 42))
    
    # Save to output path
    filtered_df.to_csv(output_path, index=False)
    print(f"Saved OOD test set with {len(filtered_df)} examples to {output_path}")
    
    return output_path