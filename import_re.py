import re
import random
import json
import os
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
import threading

class EnhancedLearningQABot:
    def __init__(self):
        self.name = "Liam"
        self.conversation_history = []
        self.word_frequencies = defaultdict(int)
        self.response_feedback = defaultdict(list)
        self.subject_expertise = defaultdict(int)
        self.question_patterns = {}
        self.knowledge_base = {}
        self.learned_responses = defaultdict(list)  # New: Store learned responses
        self.context_memory = []  # New: Remember conversation context
        self.user_preferences = {}  # New: Learn user preferences
        self.question_similarity_cache = {}  # New: Cache similar questions
        self.dynamic_keywords = defaultdict(set)  # New: Learn new keywords
        self.response_templates = defaultdict(list)  # New: Learn response patterns
        self.correction_memory = defaultdict(list)  # New: Remember corrections
        self.success_patterns = defaultdict(list)  # New: Remember successful responses
        self.all_keywords = set()  # New: Store all unique keywords
        
        # Initialize comprehensive knowledge base
        self.initialize_knowledge_base()
        
        # Enhanced subject classification keywords (will grow over time)
        self.subject_keywords = {
            'mathematics': {'math', 'calculate', 'equation', 'solve', 'number', 'algebra', 'geometry', 'calculus', 'statistics', 'probability', 'arithmetic', 'formula', 'theorem', 'plus', 'minus', 'times', 'divided'},
            'science': {'science', 'chemistry', 'physics', 'biology', 'experiment', 'theory', 'molecule', 'atom', 'cell', 'energy', 'force', 'gravity', 'evolution', 'DNA', 'periodic', 'element', 'compound'},
            'history': {'history', 'historical', 'war', 'ancient', 'civilization', 'empire', 'revolution', 'century', 'timeline', 'culture', 'dynasty', 'medieval', 'renaissance', 'battle', 'king', 'queen'},
            'geography': {'geography', 'country', 'capital', 'continent', 'ocean', 'mountain', 'river', 'climate', 'population', 'city', 'nation', 'region', 'territory', 'map', 'location'},
            'literature': {'literature', 'book', 'author', 'novel', 'poem', 'poetry', 'story', 'character', 'plot', 'theme', 'writing', 'Shakespeare', 'classic', 'chapter', 'verse'},
            'technology': {'technology', 'computer', 'software', 'internet', 'AI', 'programming', 'code', 'algorithm', 'data', 'digital', 'cyber', 'innovation', 'hardware', 'app', 'website'},
            'health': {'health', 'medicine', 'disease', 'treatment', 'doctor', 'hospital', 'symptom', 'cure', 'therapy', 'nutrition', 'exercise', 'wellness', 'medical', 'healthy', 'fitness'},
            'sports': {'sport', 'game', 'team', 'player', 'tournament', 'championship', 'football', 'basketball', 'soccer', 'tennis', 'olympic', 'athletic', 'competition', 'match', 'score'},
            'art': {'art', 'painting', 'sculpture', 'artist', 'museum', 'gallery', 'drawing', 'creative', 'design', 'color', 'masterpiece', 'renaissance', 'modern', 'canvas', 'brush'},
            'music': {'music', 'song', 'singer', 'instrument', 'melody', 'rhythm', 'composer', 'orchestra', 'band', 'concert', 'album', 'genre', 'classical', 'note', 'harmony'},
            'philosophy': {'philosophy', 'philosopher', 'ethics', 'morality', 'existence', 'consciousness', 'logic', 'reason', 'truth', 'wisdom', 'thought', 'belief', 'virtue', 'justice'},
            'economics': {'economics', 'economy', 'market', 'trade', 'business', 'finance', 'money', 'investment', 'profit', 'GDP', 'inflation', 'recession', 'commerce', 'stock', 'bank'},
            'psychology': {'psychology', 'behavior', 'mind', 'brain', 'mental', 'emotion', 'personality', 'cognitive', 'therapy', 'development', 'learning', 'memory', 'stress', 'anxiety'},
            'language': {'language', 'grammar', 'vocabulary', 'translate', 'pronunciation', 'dialect', 'linguistics', 'communication', 'speech', 'writing', 'meaning', 'word', 'sentence'}
        }
        
        # Load saved data
        self.load_ml_data()
    
    def initialize_knowledge_base(self):
        """Initialize comprehensive knowledge base across subjects"""
        self.knowledge_base = {
            'mathematics': {
                'basic_operations': {
                    'addition': 'Adding numbers together. Example: 2 + 3 = 5',
                    'subtraction': 'Taking one number away from another. Example: 5 - 2 = 3',
                    'multiplication': 'Repeated addition. Example: 3 × 4 = 12',
                    'division': 'Splitting into equal parts. Example: 12 ÷ 3 = 4'
                },
                'algebra': {
                    'variables': 'Letters representing unknown numbers (x, y, z)',
                    'equations': 'Mathematical statements with equal signs',
                    'quadratic_formula': 'x = (-b ± √(b²-4ac)) / 2a for ax² + bx + c = 0'
                },
                'geometry': {
                    'circle_area': 'Area = π × r² where r is radius',
                    'triangle_area': 'Area = ½ × base × height',
                    'pythagorean_theorem': 'a² + b² = c² for right triangles'
                }
            },
            'science': {
                'physics': {
                    'gravity': 'Force that attracts objects toward each other. Earth\'s gravity is 9.8 m/s²',
                    'speed_of_light': '299,792,458 meters per second in vacuum',
                    'newton_laws': 'Three laws of motion describing relationship between forces and motion',
                    'energy_conservation': 'Energy cannot be created or destroyed, only transformed'
                },
                'chemistry': {
                    'periodic_table': 'Organized chart of all chemical elements by atomic number',
                    'water_formula': 'H₂O - two hydrogen atoms bonded to one oxygen atom',
                    'ph_scale': 'Measures acidity/alkalinity from 0-14, 7 is neutral',
                    'atomic_structure': 'Atoms have protons, neutrons in nucleus, electrons in shells'
                },
                'biology': {
                    'cell_theory': 'All living things are made of cells, basic unit of life',
                    'DNA': 'Genetic material containing instructions for all living organisms',
                    'evolution': 'Process by which species change over time through natural selection',
                    'photosynthesis': 'Plants convert sunlight, CO₂, and water into glucose and oxygen'
                }
            },
            'history': {
                'ancient': {
                    'egypt': 'Ancient civilization along Nile River, built pyramids, ruled by pharaohs',
                    'rome': 'Powerful empire from 27 BC to 476 AD, influenced law, government, architecture',
                    'greece': 'Birthplace of democracy, philosophy, theater, and Olympic Games'
                },
                'modern': {
                    'world_war_2': 'Global conflict 1939-1945, involved most nations, ended with Allied victory',
                    'industrial_revolution': 'Period of major technological advancement, began in Britain 1760s',
                    'renaissance': 'Cultural rebirth in Europe 14th-17th centuries, art and learning flourished'
                }
            },
            'geography': {
                'continents': {
                    'asia': 'Largest continent, home to China, India, Japan, and many other countries',
                    'africa': 'Second largest continent, birthplace of humanity, diverse cultures',
                    'europe': 'Small but influential continent, many developed nations',
                    'north_america': 'Includes USA, Canada, Mexico, and Central America',
                    'south_america': 'Home to Amazon rainforest, Andes mountains',
                    'australia': 'Smallest continent, also a country, unique wildlife',
                    'antarctica': 'Southernmost continent, covered in ice, no permanent residents'
                },
                'capitals': {
                    'france': 'Paris', 'japan': 'Tokyo', 'brazil': 'Brasília', 'australia': 'Canberra',
                    'canada': 'Ottawa', 'germany': 'Berlin', 'italy': 'Rome', 'spain': 'Madrid',
                    'russia': 'Moscow', 'china': 'Beijing', 'india': 'New Delhi', 'uk': 'London'
                }
            },
            'technology': {
                'programming': {
                    'python': 'High-level programming language, great for beginners and professionals',
                    'javascript': 'Programming language for web development, runs in browsers',
                    'html': 'Markup language for creating web pages and applications',
                    'algorithm': 'Step-by-step procedure for solving problems or performing tasks'
                },
                'internet': {
                    'www': 'World Wide Web, system of interlinked hypertext documents',
                    'email': 'Electronic mail, method of exchanging digital messages',
                    'social_media': 'Online platforms for sharing content and connecting with others'
                }
            },
            'health': {
                'nutrition': {
                    'vitamins': 'Essential nutrients needed in small amounts for proper body function',
                    'protein': 'Nutrients that build and repair tissues, found in meat, beans, nuts',
                    'carbohydrates': 'Body\'s main source of energy, found in grains, fruits, vegetables',
                    'water': 'Essential for life, adults should drink about 8 glasses per day'
                },
                'exercise': {
                    'cardio': 'Exercise that increases heart rate, improves cardiovascular health',
                    'strength_training': 'Exercise using resistance to build muscle strength',
                    'flexibility': 'Range of motion in joints, improved through stretching'
                }
            },
            'programming': {
                'python': 'Python is a versatile, high-level programming language known for its readability and broad library support. It is widely used in web development, data science, automation, and AI.',
                'javascript': 'JavaScript is a dynamic scripting language primarily used for interactive web development. It runs in browsers and is essential for front-end development.',
                'variables': 'Variables are used to store data values. In Python, you declare a variable by assigning a value to a name, e.g., x = 5.',
                'functions': 'Functions are reusable blocks of code that perform a specific task. In Python, you define a function using the def keyword.',
                'loops': 'Loops are used to repeat a block of code multiple times. Common types are for-loops and while-loops.',
                'conditionals': 'Conditionals (if, elif, else) allow you to execute code based on certain conditions.',
                'oop': 'Object-Oriented Programming (OOP) is a paradigm based on objects and classes. It helps organize code and promote reuse.'
            }
        }
    
    def save_ml_data(self):
        """Save enhanced machine learning data"""
        try:
            ml_data = {
                'word_frequencies': dict(self.word_frequencies),
                'response_feedback': dict(self.response_feedback),
                'subject_expertise': dict(self.subject_expertise),
                'question_patterns': self.question_patterns,
                'conversation_history': self.conversation_history[-100:],  # Keep more history
                'learned_responses': dict(self.learned_responses),
                'context_memory': self.context_memory[-50:],
                'user_preferences': self.user_preferences,
                'question_similarity_cache': self.question_similarity_cache,
                'dynamic_keywords': {k: list(v) for k, v in self.dynamic_keywords.items()},
                'response_templates': dict(self.response_templates),
                'correction_memory': dict(self.correction_memory),
                'success_patterns': dict(self.success_patterns),
                'all_keywords': list(self.all_keywords)  # New: Save all keywords
            }
            with open('enhanced_qa_ml_data.json', 'w') as f:
                json.dump(ml_data, f, indent=2)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_ml_data(self):
        """Load enhanced machine learning data"""
        try:
            if os.path.exists('enhanced_qa_ml_data.json'):
                with open('enhanced_qa_ml_data.json', 'r') as f:
                    ml_data = json.load(f)
                self.word_frequencies = defaultdict(int, ml_data.get('word_frequencies', {}))
                self.response_feedback = defaultdict(list, ml_data.get('response_feedback', {}))
                self.subject_expertise = defaultdict(int, ml_data.get('subject_expertise', {}))
                self.question_patterns = ml_data.get('question_patterns', {})
                self.conversation_history = ml_data.get('conversation_history', [])
                self.learned_responses = defaultdict(list, ml_data.get('learned_responses', {}))
                self.context_memory = ml_data.get('context_memory', [])
                self.user_preferences = ml_data.get('user_preferences', {})
                self.question_similarity_cache = ml_data.get('question_similarity_cache', {})
                
                # Convert dynamic keywords back to sets
                dynamic_kw = ml_data.get('dynamic_keywords', {})
                self.dynamic_keywords = defaultdict(set)
                for k, v in dynamic_kw.items():
                    self.dynamic_keywords[k] = set(v)
                
                self.response_templates = defaultdict(list, ml_data.get('response_templates', {}))
                self.correction_memory = defaultdict(list, ml_data.get('correction_memory', {}))
                self.success_patterns = defaultdict(list, ml_data.get('success_patterns', {}))
                self.all_keywords = set(ml_data.get('all_keywords', []))  # New: Load all keywords
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def tokenize(self, text):
        """Advanced tokenization with better preprocessing"""
        text = re.sub(r'[^\w\s\?\!]', ' ', text.lower())
        tokens = [word for word in text.split() if len(word) > 2]
        
        # Update word frequencies for learning
        for token in tokens:
            self.word_frequencies[token] += 1
        
        return tokens
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using word overlap"""
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)
    
    def find_similar_questions(self, question: str, threshold: float = 0.3) -> List[Dict]:
        """Find similar questions from conversation history"""
        similar_questions = []
        
        for conv in self.conversation_history:
            similarity = self.calculate_text_similarity(question, conv['question'])
            if similarity >= threshold:
                similar_questions.append({
                    'question': conv['question'],
                    'response': conv['response'],
                    'similarity': similarity,
                    'feedback': conv.get('feedback', 0),
                    'subject': conv.get('subject', 'general')
                })
        
        return sorted(similar_questions, key=lambda x: x['similarity'], reverse=True)
    
    def learn_from_context(self, current_question: str):
        """Learn from conversation context"""
        if len(self.context_memory) > 0:
            # Add current question to context
            self.context_memory.append({
                'question': current_question,
                'timestamp': datetime.now().isoformat(),
                'tokens': self.tokenize(current_question)
            })
            
            # Keep only recent context (last 10 interactions)
            self.context_memory = self.context_memory[-10:]
            
            # Learn patterns from context
            if len(self.context_memory) >= 2:
                self.identify_conversation_patterns()
    
    def identify_conversation_patterns(self):
        """Identify patterns in conversation flow"""
        if len(self.context_memory) < 2:
            return
        
        # Look for topic transitions
        for i in range(1, len(self.context_memory)):
            prev_tokens = set(self.context_memory[i-1]['tokens'])
            curr_tokens = set(self.context_memory[i]['tokens'])
            
            # Find common themes
            common_tokens = prev_tokens.intersection(curr_tokens)
            if len(common_tokens) > 1:
                pattern_key = '-'.join(sorted(common_tokens)[:3])
                if pattern_key not in self.question_patterns:
                    self.question_patterns[pattern_key] = []
                self.question_patterns[pattern_key].append({
                    'from': self.context_memory[i-1]['question'],
                    'to': self.context_memory[i]['question'],
                    'timestamp': datetime.now().isoformat()
                })
    
    def learn_dynamic_keywords(self, question: str, subject: str):
        """Learn new keywords for subjects from questions"""
        tokens = self.tokenize(question)
        
        # Ensure 'general' is always present in subject_keywords
        if 'general' not in self.subject_keywords:
            self.subject_keywords['general'] = set()
        # Add new keywords to subject, handle missing subject gracefully
        if subject not in self.subject_keywords:
            self.subject_keywords[subject] = set()
        for token in tokens:
            if token not in self.subject_keywords[subject]:
                self.dynamic_keywords[subject].add(token)
        
        # Merge dynamic keywords with static ones periodically
        if len(self.dynamic_keywords[subject]) > 5:
            # Add most frequent dynamic keywords to permanent keywords
            frequent_keywords = []
            for keyword in self.dynamic_keywords[subject]:
                if self.word_frequencies[keyword] > 3:
                    frequent_keywords.append(keyword)
            
            if frequent_keywords:
                self.subject_keywords[subject].update(frequent_keywords[:3])
                # Remove learned keywords from dynamic set
                for kw in frequent_keywords:
                    self.dynamic_keywords[subject].discard(kw)
    
    def classify_subject(self, question):
        """Enhanced subject classification with learning"""
        words = self.tokenize(question)
        subject_scores = {}
        
        for subject, keywords in self.subject_keywords.items():
            score = 0
            
            # Static keywords
            for word in words:
                if word in keywords:
                    score += 3
                # Partial matching
                for keyword in keywords:
                    if word in keyword or keyword in word:
                        score += 1
            
            # Dynamic keywords
            for word in words:
                if word in self.dynamic_keywords[subject]:
                    score += 2
            
            # Historical success boost
            if subject in self.subject_expertise:
                score += self.subject_expertise[subject] * 0.1
            
            # Context boost - if recent questions were about this subject
            recent_subjects = [ctx.get('subject', 'general') for ctx in self.context_memory[-3:]]
            if subject in recent_subjects:
                score += 1
            
            subject_scores[subject] = score
        
        # Return subject with highest score
        if subject_scores:
            best_subject = max(subject_scores.items(), key=lambda x: x[1])[0]
            if subject_scores[best_subject] > 0:
                return best_subject
        
        return 'general'
    
    def extract_question_keywords(self, question):
        """Extract key terms from question for knowledge retrieval and store them in all_keywords"""
        words = self.tokenize(question)
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'does', 'can', 'will', 'would', 'should', 'could', 'tell', 'explain', 'describe'}
        keywords = [word for word in words if word not in question_words and len(word) > 2]
        # Store keywords in all_keywords
        self.all_keywords.update(keywords)
        return keywords
    
    def search_knowledge_base(self, subject, keywords) -> Optional[Tuple[str, str, str]]:
        """Enhanced knowledge base search with learning"""
        if subject not in self.knowledge_base:
            return None
        
        subject_kb = self.knowledge_base[subject]
        best_match = None
        best_score = 0
        
        def search_recursive(data, path=""):
            nonlocal best_match, best_score
            
            if isinstance(data, dict):
                for key, value in data.items():
                    key_score = sum(1 for keyword in keywords if keyword in key.lower() or key.lower() in keyword)
                    
                    if isinstance(value, str):
                        content_score = sum(1 for keyword in keywords if keyword in value.lower())
                        
                        # Boost score for learned successful patterns
                        pattern_key = f"{subject}_{key}"
                        if pattern_key in self.success_patterns:
                            content_score += len(self.success_patterns[pattern_key]) * 0.5
                        
                        total_score = key_score * 2 + content_score
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_match = (key, value, path)
                    else:
                        search_recursive(value, f"{path}/{key}" if path else key)
        
        search_recursive(subject_kb)
        return best_match
    
    def search_learned_responses(self, question: str, subject: str) -> Optional[str]:
        """Search through learned responses for similar questions"""
        if subject not in self.learned_responses:
            return None
        
        best_response = None
        best_score = 0
        
        for learned_item in self.learned_responses[subject]:
            similarity = self.calculate_text_similarity(question, learned_item['question'])
            
            # Weight by feedback score
            feedback_weight = learned_item.get('avg_feedback', 3) / 5.0
            weighted_score = similarity * feedback_weight
            
            if weighted_score > best_score and similarity > 0.4:
                best_score = weighted_score
                best_response = learned_item['response']
        
        return best_response
    
    def calculate_math_expression(self, expression):
        """Safely evaluate mathematical expressions"""
        try:
            expression = expression.replace(' ', '')
            allowed_chars = set('0123456789+-*/.()^')
            if not all(c in allowed_chars for c in expression):
                return None
            
            expression = expression.replace('^', '**')
            result = eval(expression, {"__builtins__": {}}, {})
            return result
        except:
            return None
    
    def generate_math_response(self, question):
        """Generate response for math questions with learning"""
        # Check for learned math patterns first
        learned_response = self.search_learned_responses(question, 'mathematics')
        if learned_response:
            return learned_response
        
        # Extract mathematical expressions
        math_patterns = [
            r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)',
            r'(\d+)\s*\^\s*(\d+)',
            r'sqrt\((\d+)\)',
            r'(\d+)!'
        ]
        
        for pattern in math_patterns:
            match = re.search(pattern, question)
            if match:
                if '+' in question or 'add' in question:
                    nums = re.findall(r'\d+(?:\.\d+)?', question)
                    if len(nums) >= 2:
                        result = sum(float(num) for num in nums)
                        return f"The sum is {result}"
                
                elif '-' in question or 'subtract' in question:
                    nums = re.findall(r'\d+(?:\.\d+)?', question)
                    if len(nums) >= 2:
                        result = float(nums[0]) - float(nums[1])
                        return f"The difference is {result}"
                
                elif '*' in question or 'multiply' in question:
                    nums = re.findall(r'\d+(?:\.\d+)?', question)
                    if len(nums) >= 2:
                        result = float(nums[0]) * float(nums[1])
                        return f"The product is {result}"
                
                elif '/' in question or 'divide' in question:
                    nums = re.findall(r'\d+(?:\.\d+)?', question)
                    if len(nums) >= 2 and float(nums[1]) != 0:
                        result = float(nums[0]) / float(nums[1])
                        return f"The quotient is {result}"
        
        # Try to evaluate full expression
        expression = re.search(r'[\d+\-*/().^]+', question)
        if expression:
            result = self.calculate_math_expression(expression.group())
            if result is not None:
                return f"The answer is {result}"
        
        return None
    
    def generate_response(self, question):
        """Generate comprehensive response with enhanced learning. Returns (response, is_fallback)"""
        # Learn from context
        self.learn_from_context(question)
        subject = self.classify_subject(question)
        self.learn_dynamic_keywords(question, subject)
        self.subject_expertise[subject] += 1
        keywords = self.extract_question_keywords(question)
        similar_questions = self.find_similar_questions(question)
        if similar_questions:
            best_similar = similar_questions[0]
            if best_similar['similarity'] > 0.7 and (best_similar.get('feedback', 0) or 0) >= 4:
                response = f"Based on a similar question I answered before: {best_similar['response']}"
                return self.format_response(response, subject, keywords, is_learned=True), False
        learned_response = self.search_learned_responses(question, subject)
        if learned_response:
            return self.format_response(learned_response, subject, keywords, is_learned=True), False
        if subject == 'mathematics':
            math_response = self.generate_math_response(question)
            if math_response:
                return self.format_response(math_response, subject, keywords), False
        knowledge_result = self.search_knowledge_base(subject, keywords)
        if isinstance(knowledge_result, tuple) and len(knowledge_result) == 3:
            key, content, path = knowledge_result
            response = f"Regarding {key.replace('_', ' ')}: {content}"
            related_info = self.get_related_info(subject, key)
            if related_info:
                response += f"\n\nRelated information: {related_info}"
            pattern_key = f"{subject}_{key}"
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = []
            self.success_patterns[pattern_key].append({
                'question': question,
                'keywords': keywords,
                'timestamp': datetime.now().isoformat()
            })
            is_fallback = False
        else:
            response = self.generate_contextual_response(subject, keywords, question)
            is_fallback = True
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'subject': subject,
            'keywords': keywords,
            'response': response,
            'feedback': None,
            'similar_questions': len(similar_questions),
            'confidence_factors': self.calculate_confidence_factors(subject, keywords)
        }
        self.conversation_history.append(conversation_entry)
        if len(self.conversation_history) % 3 == 0:
            self.save_ml_data()
        return self.format_response(response, subject, keywords), is_fallback
    
    def calculate_confidence_factors(self, subject: str, keywords: List[str]) -> Dict:
        """Calculate various confidence factors"""
        return {
            'subject_expertise': self.subject_expertise.get(subject, 0),
            'keyword_familiarity': sum(self.word_frequencies.get(kw, 0) for kw in keywords),
            'recent_subject_focus': len([ctx for ctx in self.context_memory[-5:] if ctx.get('subject') == subject]),
            'average_feedback': np.mean(self.response_feedback.get(subject, [3])) if self.response_feedback.get(subject) else 3
        }
    
    def get_related_info(self, subject, key):
        """Get related information from knowledge base"""
        if subject in self.knowledge_base:
            subject_kb = self.knowledge_base[subject]
            
            related = []
            for category, items in subject_kb.items():
                if isinstance(items, dict):
                    for item_key, item_value in items.items():
                        if item_key != key and len(related) < 2:
                            related.append(f"{item_key.replace('_', ' ')}: {item_value}")
            
            return " | ".join(related) if related else None
        return None
    
    def generate_contextual_response(self, subject, keywords, question):
        """Generate contextual response with learning"""
        # Check for learned response templates
        if subject in self.response_templates and self.response_templates[subject]:
            template = random.choice(self.response_templates[subject])
            return template.format(keywords=', '.join(keywords[:3]) if keywords else 'various topics')
        
        # Default responses with context awareness
        responses = {
            'mathematics': [
                "I can help with math problems! I've been learning from our conversations about calculations, formulas, and mathematical concepts.",
                "For math questions, I can solve equations, explain concepts, or help with calculations. What specific math topic interests you?",
                "Mathematics is a vast field. Based on our previous discussions, I can help with {keywords} and related topics."
            ],
            'science': [
                "Science covers many areas like physics, chemistry, and biology. I've been learning about {keywords} from our conversations.",
                "I'd be happy to explain scientific concepts! My knowledge grows with each question you ask.",
                "Science is fascinating! I've noticed you're interested in {keywords}. What would you like to explore further?"
            ],
            'general': [
                "I'm continuously learning from our conversations! I can help with questions across many subjects.",
                "That's an interesting question about {keywords}! Let me think about what I've learned that might help.",
                "I'm here to help with information across various subjects. My knowledge base grows with each interaction!"
            ]
        }
        
        subject_responses = responses.get(subject, responses['general'])
        base_response = random.choice(subject_responses)
        
        # Add keyword context
        if keywords:
            base_response = base_response.format(keywords=', '.join(keywords[:3]))
        else:
            base_response = base_response.replace(" {keywords}", "").replace("{keywords}", "various topics")
        
        return base_response
    
    def format_response(self, response, subject, keywords, is_learned=False):
        """Format response with enhanced context and confidence"""
        confidence_factors = self.calculate_confidence_factors(subject, keywords)
        
        # Calculate overall confidence
        base_confidence = min(100, max(20, confidence_factors['subject_expertise'] * 5 + 50))
        keyword_boost = min(20, confidence_factors['keyword_familiarity'] * 2)
        context_boost = confidence_factors['recent_subject_focus'] * 5
        feedback_boost = (confidence_factors['average_feedback'] - 3) * 10
        
        overall_confidence = min(100, max(20, base_confidence + keyword_boost + context_boost + feedback_boost))
        
        # Format response with learning indicators
        learning_indicator = "🧠 Learned" if is_learned else "📚 Knowledge"
        formatted = f"[{subject.title()}] {response}"
        
        # Add confidence and learning information
        if overall_confidence > 85:
            formatted += f"\n\n💡 {learning_indicator} | Confidence: {overall_confidence:.0f}% - I'm very confident about this topic!"
        elif overall_confidence > 70:
            formatted += f"\n\n💡 {learning_indicator} | Confidence: {overall_confidence:.0f}% - I have good knowledge of this area."
        elif overall_confidence > 50:
            formatted += f"\n\n💡 {learning_indicator} | Confidence: {overall_confidence:.0f}% - I'm learning more about this topic."
        else:
            formatted += f"\n\n💡 {learning_indicator} | Confidence: {overall_confidence:.0f}% - This is a newer topic for me."
        
        # Add learning progress if applicable
        if is_learned:
            formatted += f"\n🎯 I remembered this from our previous conversations!"
        
        return formatted
    
    def get_feedback(self, rating):
        """Enhanced feedback processing with learning"""
        if not self.conversation_history:
            return "No recent response to rate."
        
        last_interaction = self.conversation_history[-1]
        subject = last_interaction['subject']
        question = last_interaction['question']
        response = last_interaction['response']
        
        # Store feedback
        self.response_feedback[subject].append(rating)
        self.conversation_history[-1]['feedback'] = rating
        
        # Learn from feedback
        if rating >= 4:
            # Positive feedback - reinforce this response
            self.subject_expertise[subject] += 2
            
            # Store as learned response
            if subject not in self.learned_responses:
                self.learned_responses[subject] = []
            
            # Check if similar response already exists
            existing_response = None
            for learned_item in self.learned_responses[subject]:
                if self.calculate_text_similarity(question, learned_item['question']) > 0.6:
                    existing_response = learned_item
                    break
            
            if existing_response:
                # Update existing response
                existing_response['feedback_scores'].append(rating)
                existing_response['avg_feedback'] = np.mean(existing_response['feedback_scores'])
                existing_response['usage_count'] += 1
            else:
                # Add new learned response
                self.learned_responses[subject].append({
                    'question': question,
                    'response': response,
                    'feedback_scores': [rating],
                    'avg_feedback': rating,
                    'usage_count': 1,
                    'learned_date': datetime.now().isoformat(),
                    'keywords': last_interaction.get('keywords', [])
                })
            
            # Store successful response template
            if subject not in self.response_templates:
                self.response_templates[subject] = []
            
            # Extract template pattern
            template = re.sub(r'\b\d+\b', '{number}', response)
            template = re.sub(r'\b[A-Z][a-z]+\b', '{proper_noun}', template)
            if template not in self.response_templates[subject]:
                self.response_templates[subject].append(template)
        
        elif rating <= 2:
            # Negative feedback - learn what to avoid
            self.subject_expertise[subject] = max(0, self.subject_expertise[subject] - 1)
            
            # Store correction opportunity
            if subject not in self.correction_memory:
                self.correction_memory[subject] = []
            
            self.correction_memory[subject].append({
                'question': question,
                'poor_response': response,
                'rating': rating,
                'timestamp': datetime.now().isoformat(),
                'keywords': last_interaction.get('keywords', [])
            })
        
        # Learn user preferences
        self.learn_user_preferences(last_interaction, rating)
        
        # Save learning data
        self.save_ml_data()
        
        feedback_messages = {
            5: "Excellent! I'm learning that this type of response works really well.",
            4: "Great! I'll remember this successful approach for similar questions.",
            3: "Thanks for the feedback! I'll use this to improve my responses.",
            2: "I'll work on improving responses like this. Thanks for the honest feedback.",
            1: "I understand this wasn't helpful. I'm learning what to avoid for next time."
        }
        
        return f"Thank you for rating my {subject} response! {feedback_messages.get(rating, 'Thanks for the feedback!')}"
    
    def learn_user_preferences(self, interaction, rating):
        """Learn user preferences from interactions"""
        subject = interaction['subject']
        keywords = interaction.get('keywords', [])
        
        # Track preference patterns
        pref_key = f"{subject}_style"
        if pref_key not in self.user_preferences:
            self.user_preferences[pref_key] = {'detailed': 0, 'concise': 0, 'examples': 0}
        
        # Analyze response characteristics
        response_length = len(interaction['response'].split())
        has_examples = 'example' in interaction['response'].lower() or ':' in interaction['response']
        
        if rating >= 4:
            if response_length > 50:
                self.user_preferences[pref_key]['detailed'] += 1
            elif response_length < 30:
                self.user_preferences[pref_key]['concise'] += 1
            
            if has_examples:
                self.user_preferences[pref_key]['examples'] += 1
        
        # Learn keyword preferences
        if rating >= 4:
            for keyword in keywords:
                pref_key = f"keyword_{keyword}"
                self.user_preferences[pref_key] = self.user_preferences.get(pref_key, 0) + 1
    
    def teach_me(self, topic, information):
        """Allow user to teach the bot new information"""
        if not topic or not information:
            return "Please provide both a topic and the information you'd like to teach me!"
        
        # Classify the topic
        subject = self.classify_subject(topic)
        keywords = self.extract_question_keywords(topic)
        
        # Store the new information
        teaching_entry = {
            'topic': topic,
            'information': information,
            'subject': subject,
            'keywords': keywords,
            'taught_date': datetime.now().isoformat(),
            'usage_count': 0
        }
        
        # Add to learned responses
        if subject not in self.learned_responses:
            self.learned_responses[subject] = []
        
        self.learned_responses[subject].append({
            'question': topic,
            'response': information,
            'feedback_scores': [5],  # Assume user-taught info is high quality
            'avg_feedback': 5,
            'usage_count': 0,
            'learned_date': datetime.now().isoformat(),
            'keywords': keywords,
            'user_taught': True
        })
        
        # Update expertise
        self.subject_expertise[subject] += 3
        
        # Learn new keywords
        self.learn_dynamic_keywords(topic, subject)
        
        # Save the learning
        self.save_ml_data()
        
        return f"Thank you for teaching me about {topic}! I've learned: {information}\nI've categorized this under {subject} and will remember it for future questions."
    
    def forget_topic(self, topic):
        """Allow user to remove incorrect information"""
        removed_count = 0
        
        for subject in self.learned_responses:
            self.learned_responses[subject] = [
                item for item in self.learned_responses[subject]
                if self.calculate_text_similarity(topic, item['question']) < 0.5
            ]
            removed_count += len(self.learned_responses[subject])
        
        # Also remove from correction memory
        for subject in self.correction_memory:
            original_length = len(self.correction_memory[subject])
            self.correction_memory[subject] = [
                item for item in self.correction_memory[subject]
                if self.calculate_text_similarity(topic, item['question']) < 0.5
            ]
            removed_count += original_length - len(self.correction_memory[subject])
        
        self.save_ml_data()
        
        if removed_count > 0:
            return f"I've removed {removed_count} learned responses related to '{topic}'."
        else:
            return f"I couldn't find any learned responses related to '{topic}' to remove."
    
    def show_expertise(self):
        """Show current subject expertise levels with learning details"""
        if not self.subject_expertise:
            return "I haven't answered any questions yet!"
        
        expertise_str = "🧠 My Current Learning Progress:\n" + "="*50 + "\n"
        sorted_subjects = sorted(self.subject_expertise.items(), key=lambda x: x[1], reverse=True)
        
        for subject, level in sorted_subjects:
            bars = "█" * min(level, 20)
            learned_count = len(self.learned_responses.get(subject, []))
            avg_feedback = np.mean(self.response_feedback.get(subject, [3])) if self.response_feedback.get(subject) else 3
            
            expertise_str += f"{subject.title()}: {bars} ({level})\n"
            expertise_str += f"  📚 Learned responses: {learned_count}\n"
            expertise_str += f"  ⭐ Average feedback: {avg_feedback:.1f}/5.0\n"
            expertise_str += f"  🎯 Dynamic keywords: {len(self.dynamic_keywords.get(subject, []))}\n\n"
        
        return expertise_str
    
    def show_learned_responses(self):
        """Show learned responses summary"""
        if not self.learned_responses:
            return "I haven't learned any specific responses yet!"
        
        summary = "🎓 My Learned Knowledge:\n" + "="*40 + "\n"
        
        for subject, responses in self.learned_responses.items():
            if responses:
                summary += f"\n📖 {subject.title()} ({len(responses)} learned responses):\n"
                
                # Show top 3 best responses
                sorted_responses = sorted(responses, key=lambda x: x['avg_feedback'], reverse=True)
                for i, response in enumerate(sorted_responses[:3]):
                    summary += f"  {i+1}. Topic: {response['question'][:50]}...\n"
                    summary += f"     Rating: {response['avg_feedback']:.1f}/5.0 | Used: {response['usage_count']} times\n"
                    if response.get('user_taught'):
                        summary += f"     👨‍🏫 User taught\n"
                    summary += "\n"
        
        return summary
    
    def show_stats(self):
        """Show comprehensive statistics with learning metrics"""
        total_questions = len(self.conversation_history)
        if total_questions == 0:
            return "No questions answered yet!"
        
        # Calculate learning metrics
        total_learned = sum(len(responses) for responses in self.learned_responses.values())
        total_corrections = sum(len(corrections) for corrections in self.correction_memory.values())
        
        # Subject distribution
        subjects = [conv['subject'] for conv in self.conversation_history]
        subject_counts = Counter(subjects)
        
        # Average feedback
        feedbacks = [conv['feedback'] for conv in self.conversation_history if conv.get('feedback')]
        avg_feedback = sum(feedbacks) / len(feedbacks) if feedbacks else 0
        
        # Learning rate (how many responses become learned)
        learning_rate = (total_learned / total_questions * 100) if total_questions > 0 else 0
        
        # Most common keywords
        all_keywords = []
        for conv in self.conversation_history:
            all_keywords.extend(conv.get('keywords', []))
        common_keywords = Counter(all_keywords).most_common(5)
        
        # Recent learning trend
        recent_feedback = [conv['feedback'] for conv in self.conversation_history[-10:] if conv.get('feedback')]
        recent_avg = sum(recent_feedback) / len(recent_feedback) if recent_feedback else 0
        
        stats = f"""
🧠 Enhanced Learning Bot Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 BASIC METRICS:
  📝 Total Questions Answered: {total_questions}
  ⭐ Average Rating: {avg_feedback:.1f}/5.0
  📈 Recent Performance: {recent_avg:.1f}/5.0 (last 10 responses)
  🧠 Knowledge Areas: {len(subject_counts)}
  📚 Vocabulary Size: {len(self.word_frequencies)}

🎓 LEARNING METRICS:
  🧠 Learned Responses: {total_learned}
  📈 Learning Rate: {learning_rate:.1f}% (responses that became learned)
  🔄 Corrections Stored: {total_corrections}
  🎯 Dynamic Keywords: {sum(len(kw) for kw in self.dynamic_keywords.values())}
  💭 Context Memory: {len(self.context_memory)} recent interactions

📈 Subject Distribution:
{chr(10).join([f"  {subject.title()}: {count} questions" for subject, count in subject_counts.most_common()])}

🔤 Most Common Keywords:
{chr(10).join([f"  {keyword}: {count} times" for keyword, count in common_keywords])}

🎯 Expertise Levels:
{chr(10).join([f"  {subject.title()}: Level {level}" for subject, level in sorted(self.subject_expertise.items(), key=lambda x: x[1], reverse=True)])}

📊 Learning Quality:
{chr(10).join([f"  {subject.title()}: {np.mean(feedback):.1f}/5.0 avg rating" for subject, feedback in self.response_feedback.items() if feedback])}
"""
        return stats
    
    def remember_fact(self, fact):
        """Allow user to store a free-form fact in memory."""
        if not fact:
            return "Please provide something for me to remember!"
        subject = 'general'
        keywords = self.extract_question_keywords(fact)
        if subject not in self.learned_responses:
            self.learned_responses[subject] = []
        self.learned_responses[subject].append({
            'question': fact,
            'response': fact,
            'feedback_scores': [5],
            'avg_feedback': 5,
            'usage_count': 0,
            'learned_date': datetime.now().isoformat(),
            'keywords': keywords,
            'user_taught': True
        })
        self.save_ml_data()
        return f"I've remembered: '{fact}'!"

    def store_learned_qa(self, question, answer):
        """Store a Q&A pair in learned_responses for future recall."""
        subject = self.classify_subject(question)
        keywords = self.extract_question_keywords(question)
        if subject not in self.learned_responses:
            self.learned_responses[subject] = []
        # Avoid duplicates
        for item in self.learned_responses[subject]:
            if self.calculate_text_similarity(question, item['question']) > 0.7:
                return  # Already learned
        self.learned_responses[subject].append({
            'question': question,
            'response': answer,
            'feedback_scores': [5],
            'avg_feedback': 5,
            'usage_count': 1,
            'learned_date': datetime.now().isoformat(),
            'keywords': keywords,
            'user_taught': False
        })
        self.save_ml_data()

    def ask_gemini(self, prompt):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Gemini API key not found. Please set GEMINI_API_KEY environment variable."
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        system_prompt = (
            "You are Liam, a chat bot designed to assist users with a wide range of topics. "
            
        )
        full_prompt = f"{system_prompt}\n\nUser question: {prompt}"
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(full_prompt)
        # The latest google-generativeai returns response.text or response.candidates[0].text
        if hasattr(response, 'text'):
            return response.text.strip()
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].text.strip()
        else:
            return str(response)

    def chat(self):
        """Enhanced chat interface with learning commands and memory."""
        print(f"🤖 Welcome to {self.name}! I'm an AI that learns from our conversations.")
        print("🧠 I remember previous questions, learn from your feedback, and improve over time!")
        print("📚 Subjects I know: Math, Science, History, Geography, Technology, Health, and more!")
        print("\n💬 Available Commands:")
        print("  • 'quit' - Exit the chat")
        print("  • 'stats' - Show detailed statistics and learning metrics")
        print("  • 'expertise' - Show my current knowledge levels")
        print("  • 'learned' - Show my learned responses")
        print("  • 'teach [topic] [information]' - Teach me something new")
        print("  • 'forget [topic]' - Remove incorrect information")
        print("  • 'rate X' - Rate my last response (1-5)")
        print("  • 'remember this: [fact]' - Remember a fact for later recall")
        print("  • 'keywords' - Show all stored keywords")
        print("\n⭐ Rate my answers to help me learn what works best!")
        print("=" * 80)
        last_answer = None
        last_question = None
        while True:
            user_input = input(f"\n🤔 Ask me anything (or use a command): ").strip()

            if user_input.lower() == 'quit':
                print(f"\n👋 {self.name}: Thank you for helping me learn! I'll remember our conversation.")
                break

            if user_input.lower() == 'stats':
                print(f"\n📊 {self.name}: {self.show_stats()}")
                continue

            if user_input.lower() == 'expertise':
                print(f"\n🧠 {self.name}: {self.show_expertise()}")
                continue

            if user_input.lower() == 'learned':
                print(f"\n🎓 {self.name}: {self.show_learned_responses()}")
                continue

            if user_input.lower().startswith('teach '):
                parts = user_input[6:].split(' ', 1)
                if len(parts) >= 2:
                    topic, info = parts[0], parts[1]
                    print(f"\n👨‍🏫 {self.name}: {self.teach_me(topic, info)}")
                else:
                    print(f"\n❌ {self.name}: Please use format: teach [topic] [information]")
                continue

            if user_input.lower().startswith('forget '):
                topic = user_input[7:]
                print(f"\n🗑️ {self.name}: {self.forget_topic(topic)}")
                continue

            if user_input.lower().startswith('rate '):
                try:
                    rating = int(user_input.split()[1])
                    if 1 <= rating <= 5:
                        print(f"\n✅ {self.name}: {self.get_feedback(rating)}")
                    else:
                        print(f"\n❌ {self.name}: Please rate between 1-5.")
                except:
                    print(f"\n❌ {self.name}: Invalid rating. Use 'rate X' where X is 1-5.")
                continue

            if user_input.lower().startswith('remember this:'):
                fact = user_input[14:].strip()
                print(f"\n💾 {self.name}: {self.remember_fact(fact)}")
                continue

            if user_input.lower() == 'keywords':
                print(f"\n🔑 {self.name}: Stored keywords ({len(self.all_keywords)}):\n" + ', '.join(sorted(self.all_keywords)))
                continue

            # Check for learned answer before calling Gemini
            subject = self.classify_subject(user_input)
            learned_answer = self.search_learned_responses(user_input, subject)
            if learned_answer:
                print(f"\n🤖 {self.name}: [From memory] {learned_answer}")
                last_answer = learned_answer
                last_question = user_input
            else:
                # Use generate_response and fallback to Gemini if needed
                if user_input:
                    response, is_fallback = self.generate_response(user_input)
                    if not is_fallback:
                        print(f"\n🤖 {self.name}: {response}")
                        last_answer = response
                        last_question = user_input
                    else:
                        gemini_answer = self.ask_gemini(user_input)
                        print(f"\n🤖 {self.name}: {gemini_answer}")
                        last_answer = gemini_answer
                        last_question = user_input
                        self.store_learned_qa(user_input, gemini_answer)
                        print("\n💭 How was my answer? Type 'rate X' (1-5) to help me learn!")

            # Always ask Gemini in the background and store the answer for training
            def background_gemini_train(question):
                gemini_answer = self.ask_gemini(question)
                self.store_learned_qa(question, gemini_answer)
            threading.Thread(target=background_gemini_train, args=(user_input,), daemon=True).start()

# Run the enhanced chatbot
if __name__ == "__main__":
    try:
        import numpy as np
    except ImportError:
        print("Installing numpy...")
        os.system("pip install numpy")
        import numpy as np
    
    bot = EnhancedLearningQABot()
    bot.chat()