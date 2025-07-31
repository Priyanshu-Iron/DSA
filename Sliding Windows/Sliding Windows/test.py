import os
import re
import sklearn
import imblearn
import pandas as pd
import pdfplumber
from lxml import etree
import torch
import logging
import gc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import warnings
import time
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_recall_curve

class GPUMonitor:
    """GPU monitoring for training"""
    def __init__(self, device):
        self.device = device
        self.is_cuda = device.type == 'cuda'
        
    def log_memory_usage(self, step_name=""):
        """Log current GPU memory usage"""
        if self.is_cuda:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            logging.info(f"📊 {step_name} - GPU Memory:")
            logging.info(f"  Allocated: {allocated:.2f} GB")
            logging.info(f"  Reserved: {reserved:.2f} GB") 
            logging.info(f"  Max Allocated: {max_allocated:.2f} GB")
            
            # Check if GPU is actually being used
            if allocated < 0.1:
                logging.warning("⚠️ GPU memory usage is very low - model may not be using GPU!")

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_DISABLED'] = 'true'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CUDA environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.backends.cudnn.benchmark = True

@dataclass
class Config:
    """Enhanced configuration for better GPU utilization"""
    base_path: str = '/kaggle/input/make-data-count-finding-data-references'
    output_path: str = '/kaggle/working/submission.csv'
    validation_output_path: str = '/kaggle/working/validation_results.csv'
    
    local_models_path: str = '/kaggle/input/packages-and-modules/offline_models'
    use_local_models: bool = True
    
    use_pdf: bool = True
    max_length: int = 384  # Reduced for better GPU utilization
    batch_size: int = 8    # Increased batch size
    num_epochs: int = 6    # Reduced epochs for faster training
    learning_rate: float = 2e-5
    validation_split: float = 0.2
    n_folds: int = 3

    @property
    def train_pdf_path(self):
        return f'{self.base_path}/train/PDF'
    
    @property
    def train_xml_path(self):
        return f'{self.base_path}/train/XML'
    
    @property
    def test_pdf_path(self):
        return f'{self.base_path}/test/PDF'
    
    @property
    def test_xml_path(self):
        return f'{self.base_path}/test/XML'
    
    @property
    def train_labels_path(self):
        return f'{self.base_path}/train_labels.csv'

# Global config instance
config = Config()

def setup_device():
    """Enhanced device setup for dual GPU utilization"""
    print("=" * 60)
    print("🔧 DUAL GPU SETUP AND DIAGNOSTICS")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        return torch.device('cpu')

    # Clear any existing memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    logging.info(f"🎯 AVAILABLE GPUS: {num_gpus}")
    
    # Test all available GPUs
    for i in range(num_gpus):
        try:
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            device_name = props.name
            
            logging.info(f"🎯 GPU {i}: {device_name}")
            logging.info(f"💾 GPU {i} MEMORY: {total_memory:.1f} GB")
            
            # Test GPU operations
            test_device = torch.device(f'cuda:{i}')
            test_tensor = torch.randn(1000, 1000, device=test_device)
            result = torch.matmul(test_tensor, test_tensor)
            torch.cuda.synchronize(test_device)
            
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            logging.info(f"✅ GPU {i} TEST SUCCESSFUL - Memory: {allocated:.2f} GB")
            
            # Clean up
            del test_tensor, result
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"❌ GPU {i} setup failed: {e}")
    
    # Use the first available GPU
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    logging.info(f"🎯 PRIMARY DEVICE SET TO: {device}")
    
    return device

def get_model_path(model_name):
    """Get the appropriate model path (local or online)"""
    if config.use_local_models:
        # Map online model names to local folder names
        model_mapping = {
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": "biomedbert",
            "allenai/scibert_scivocab_uncased": "scibert",
            # Add more mappings as needed
        }
        
        local_model_name = model_mapping.get(model_name, model_name.split('/')[-1])
        local_path = os.path.join(config.local_models_path, local_model_name)
        
        if os.path.exists(local_path):
            logging.info(f"✅ Using local model: {local_path}")
            return local_path
        else:
            logging.warning(f"❌ Local model not found at {local_path}, falling back to online")
            return model_name
    else:
        return model_name

def select_model():
    """Select model based on available GPU memory"""
    if device.type == 'cpu':
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        logging.info(f"🤖 SELECTED MODEL: {model_name}")
        logging.info(f"🖥️  DEVICE: CPU (No GPU available)")
        return get_model_path(model_name)
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_name = torch.cuda.get_device_properties(0).name
        
        if total_memory >= 15:
            model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            reason = "High GPU memory (≥15GB)"
        elif total_memory >= 8:
            model_name = "allenai/scibert_scivocab_uncased"
            reason = "Medium GPU memory (8-15GB)"
        else:
            model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            reason = "Low GPU memory (<8GB)"
        
        logging.info(f"🤖 SELECTED MODEL: {model_name}")
        logging.info(f"🖥️  DEVICE: {gpu_name}")
        logging.info(f"💾 GPU MEMORY: {total_memory:.1f} GB")
        logging.info(f"📋 REASON: {reason}")
        
        return get_model_path(model_name)
    except Exception as e:
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        logging.info(f"🤖 SELECTED MODEL: {model_name}")
        logging.info(f"⚠️  REASON: Fallback due to exception - {e}")
        return get_model_path(model_name)

BEST_MODEL = select_model()

def preprocess_text_for_classification(text):
    """Enhanced text preprocessing with improved feature extraction"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Clean text
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\-/:]', ' ', text)
    
    # Enhanced section patterns
    data_sections = []
    section_patterns = [
        r'(?i)data\s+availability\s*statement?[^.]*?(?:\.|$)',
        r'(?i)data\s+availability[^.]*?(?:\.|$)',
        r'(?i)supplementary\s+(?:data|material|information)[^.]*?(?:\.|$)',
        r'(?i)supporting\s+information[^.]*?(?:\.|$)',
        r'(?i)additional\s+file[s]?[^.]*?(?:\.|$)',
        r'(?i)raw\s+data[^.]*?(?:\.|$)',
        r'(?i)source\s+data[^.]*?(?:\.|$)',
        r'(?i)original\s+data[^.]*?(?:\.|$)',
        r'(?i)data\s+deposited[^.]*?(?:\.|$)',
    ]
    
    for pattern in section_patterns:
        matches = re.findall(pattern, text)
        data_sections.extend(matches)
    
    # Enhanced repository patterns
    repo_patterns = [
        r'(?i)(?:deposited|archived|uploaded|available)\s+(?:at|in|from)[^.]*?(?:\.|$)',
        r'(?i)(?:figshare|dryad|zenodo|mendeley|github|gitlab)[^.]*?(?:\.|$)',
        r'(?i)data\s+repository[^.]*?(?:\.|$)',
        r'(?i)publicly\s+available[^.]*?(?:\.|$)',
    ]
    
    for pattern in repo_patterns:
        matches = re.findall(pattern, text)
        data_sections.extend(matches)
    
    # Enhanced database patterns
    db_patterns = [
        r'(?i)(?:downloaded|obtained|retrieved|accessed)\s+from[^.]*?(?:\.|$)',
        r'(?i)(?:ncbi|geo|arrayexpress|ensembl|uniprot)[^.]*?(?:\.|$)',
        r'(?i)public\s+database[^.]*?(?:\.|$)',
        r'(?i)previously\s+published[^.]*?(?:\.|$)',
        r'(?i)reference\s+genome[^.]*?(?:\.|$)',
    ]
    
    for pattern in db_patterns:
        matches = re.findall(pattern, text)
        data_sections.extend(matches)
    
    # Prioritize data sections
    if data_sections:
        combined_text = ' '.join(data_sections[:10])
    else:
        # Enhanced keyword scoring
        sentences = re.split(r'[.!?]+', text)
        data_keywords = [
            'data', 'dataset', 'repository', 'database', 'archive', 'supplementary',
            'doi', 'url', 'http', 'available', 'deposited', 'uploaded', 'figshare',
            'dryad', 'zenodo', 'github', 'ncbi', 'geo', 'arrayexpress', 'original',
            'raw', 'source', 'publicly', 'accessed', 'retrieved'
        ]
        
        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 30:
                score = sum(2 if keyword in ['data', 'dataset', 'repository', 'original', 'raw'] 
                           else 1 for keyword in data_keywords if keyword in sentence.lower())
                if score >= 3:  # Higher threshold for relevance
                    relevant_sentences.append(sentence)
        
        combined_text = ' '.join(relevant_sentences[:10])
    
    # Ensure length constraints
    if len(combined_text.strip()) < 100:
        combined_text = text[:2000]
    elif len(combined_text) > 2000:
        combined_text = combined_text[:2000]
    
    return combined_text.strip()

class WeightedTrainer(Trainer):
    """Enhanced Trainer with improved focal loss"""
    
    def __init__(self, class_weights, alpha=0.25, gamma=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Ensure inputs are on the correct device
        inputs = {k: v.to(self.args.device, non_blocking=True) if hasattr(v, 'to') else v 
                  for k, v in inputs.items()}
        
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        weights = torch.tensor([self.class_weights[0], self.class_weights[1]], 
                             dtype=torch.float, device=labels.device)  # CRITICAL: Same device
        weighted_loss = focal_loss * weights[labels]
        
        loss = weighted_loss.mean()
        return (loss, outputs) if return_outputs else loss

class CitationDataset(Dataset):
    """Dataset class for citation data"""
    
    def __init__(self, texts, labels, tokenizer, max_length=config.max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw_text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        processed_text = preprocess_text_for_classification(raw_text)
        
        if len(processed_text.strip()) < 50:
            processed_text = raw_text[:1500] if len(raw_text) > 1500 else raw_text
            
        encoding = self.tokenizer(
            processed_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DocumentProcessor:
    """Document processor for PDF and XML files"""
    
    def __init__(self, use_pdf=config.use_pdf):
        self.use_pdf = use_pdf
        self.doc_type = "PDF" if use_pdf else "XML"
        logging.info(f"Document processor initialized for {self.doc_type} files")

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF with memory optimization"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ''
                max_pages = min(15, len(pdf.pages))
                for i in range(max_pages):
                    try:
                        page_text = pdf.pages[i].extract_text()
                        if page_text:
                            text += page_text + '\n'
                            if len(text) > 10000:
                                break
                    except Exception:
                        continue
                return text.strip()
        except Exception as e:
            logging.error(f"Failed to process PDF {file_path}: {e}")
            return ''

    def extract_text_from_xml(self, file_path):
        """Extract text from XML with improved parsing"""
        try:
            parser = etree.XMLParser(recover=True, encoding='utf-8')
            with open(file_path, 'rb') as f:
                content = f.read()
            if len(content) > 2 * 1024 * 1024:
                content = content[:2 * 1024 * 1024]
            tree = None
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    content_str = content.decode(encoding)
                    tree = etree.fromstring(content_str, parser=parser)
                    break
                except Exception:
                    continue
            if tree is None:
                return ''
            text_elements = []
            relevant_tags = ['p', 'title', 'abstract', 'body', 'section', 'div', 'text']
            for elem in tree.iter():
                if elem.tag.lower() in relevant_tags or elem.text:
                    if elem.text and elem.text.strip():
                        text_elements.append(elem.text.strip())
                    if elem.tail and elem.tail.strip():
                        text_elements.append(elem.tail.strip())
                    if len(text_elements) > 2000:
                        break
            full_text = ' '.join(text_elements)
            return full_text[:10000] if len(full_text) > 10000 else full_text
        except Exception as e:
            logging.error(f"Error parsing XML {file_path}: {e}")
            return ''

    def process_document(self, file_path):
        """Process document based on type"""
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            return ''
        if self.use_pdf and file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif not self.use_pdf and file_path.lower().endswith('.xml'):
            return self.extract_text_from_xml(file_path)
        else:
            return ''

    def get_document_path(self, article_id, is_train=True):
        """Get the appropriate document path"""
        if is_train:
            base_path = config.train_pdf_path if self.use_pdf else config.train_xml_path
        else:
            base_path = config.test_pdf_path if self.use_pdf else config.test_xml_path
        extension = '.pdf' if self.use_pdf else '.xml'
        return os.path.join(base_path, f"{article_id}{extension}")

class EnhancedDatasetDetector:
    """Enhanced dataset detection with improved patterns"""
    
    def __init__(self):
        self.doi_pattern = re.compile(
            r'(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[-.;()/:A-Za-z0-9]+)', 
            re.IGNORECASE
        )
        self.doi_pattern_short = re.compile(r'\b10\.\d{4,9}/[-.;()/:A-Za-z0-9]+\b')
        self.accession_patterns = [
            re.compile(r'\b(?:GSE|GDS|GPL|GSM)\d{4,10}\b'),
            re.compile(r'\b(?:SRA|SRR|SRX|SRS|SRP|ERR|ERX|ERS|ERP|DRR|DRX|DRS|DRP)\d{6,10}\b'),
            re.compile(r'\b(?:PRJNA|PRJEB|PRJDB)\d{6,10}\b'),
            re.compile(r'\b(?:SAMN|SAME|SAMD)\d{8,12}\b'),
            re.compile(r'\bPDB\s+[A-Z0-9]{4}\b', re.IGNORECASE),
            re.compile(r'\b[A-Z]{1,2}\d{5,8}(?:\.\d+)?\b'),
            re.compile(r'\bE-[A-Z]{4}-\d+\b'),
            re.compile(r'\bCHEMBL\d{6,10}\b'),
            re.compile(r'\bEPIISL\d{6,10}\b'),
            re.compile(r'\bEMPIAR-\d{5}\b'),
            re.compile(r'figshare\.com/[^\s]+'),
            re.compile(r'zenodo\.org/record/\d+'),
            re.compile(r'datadryad\.org/[^\s]+'),
        ]
        self.blacklist = {str(year) for year in range(1900, 2050)}
        self.blacklist.update(['10.1', '10.2', '10.3', '10.4', '10.5'])

    def detect_datasets(self, text):
        """Detect dataset references in text"""
        if not isinstance(text, str) or not text.strip():
            return set()
        datasets = set()
        for match in self.doi_pattern.finditer(text):
            doi = match.group(1)
            if self._is_valid_doi(doi):
                datasets.add(f"https://doi.org/{doi}")
        for match in self.doi_pattern_short.finditer(text):
            doi = match.group()
            if self._is_valid_doi(doi):
                datasets.add(f"https://doi.org/{doi}")
        for pattern in self.accession_patterns:
            for match in pattern.finditer(text):
                acc_id = match.group().strip()
                if self._is_valid_accession(acc_id):
                    datasets.add(acc_id)
        return datasets

    def _is_valid_doi(self, doi):
        """Validate DOI format"""
        if not doi or len(doi) < 8 or doi in self.blacklist:
            return False
        return bool(re.match(r'10\.\d{4,9}/[-.;()/:A-Za-z0-9]+', doi))

    def _is_valid_accession(self, acc_id):
        """Validate accession ID"""
        if acc_id in self.blacklist or len(acc_id) < 4 or (acc_id.isdigit() and len(acc_id) < 6):
            return False
        return True

class ScientificClassifier:
    """Enhanced scientific text classifier"""
    
    def __init__(self, model_name=BEST_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        self.label_map = {'Primary': 1, 'Secondary': 0}
        self.reverse_label_map = {1: 'Primary', 0: 'Secondary'}
        self.device = device
        self.best_threshold = 0.5
        
        logging.info(f"🔧 CLASSIFIER INITIALIZED WITH MODEL: {self.model_name}")

    def load_model(self):
        """Enhanced model loading with proper GPU setup"""
        try:
            logging.info(f"📥 LOADING MODEL: {self.model_name}")
            logging.info(f"🎯 TARGET DEVICE: {self.device}")
            
            # Clear memory before loading
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Create GPU monitor
            monitor = GPUMonitor(self.device)
            monitor.log_memory_usage("Before Model Loading")
            
            # Check if using local model
            is_local = os.path.exists(self.model_name) if isinstance(self.model_name, str) else False
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                use_fast=True,
                local_files_only=is_local
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info(f"✅ TOKENIZER LOADED")
            
            # Load model with enhanced GPU configuration
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto' if self.device.type == 'cuda' else None,
                low_cpu_mem_usage=True,
                local_files_only=is_local
            )
            
            # Explicitly move to device and verify
            self.model = self.model.to(self.device)
            model_device = next(self.model.parameters()).device
            
            logging.info(f"✅ MODEL LOADED ON DEVICE: {model_device}")
            
            # Verify model is on correct device
            if model_device != self.device:
                logging.warning(f"⚠️ Model on {model_device}, expected {self.device}")
                self.model = self.model.to(self.device)
                model_device = next(self.model.parameters()).device
                logging.info(f"🔄 Model moved to: {model_device}")
            
            # Log model statistics
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logging.info(f"📊 TOTAL PARAMETERS: {param_count:,}")
            logging.info(f"🔧 TRAINABLE PARAMETERS: {trainable_params:,}")
            
            monitor.log_memory_usage("After Model Loading")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ FAILED TO LOAD MODEL {self.model_name}: {e}")
            return False

    def get_training_args(self, batch_size):
        """Enhanced training arguments with proper GPU settings"""
        return TrainingArguments(
            output_dir='./results',
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Reduced for better GPU utilization
            warmup_steps=200,  # Reduced warmup
            weight_decay=0.01,
            learning_rate=config.learning_rate,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,  # More frequent evaluation
            save_strategy="steps", 
            save_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=True,  # Always use fp16 for GPU
            dataloader_pin_memory=True,  # Enable for better GPU transfer
            dataloader_num_workers=2,    # Use some workers for data loading
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to=None,
            save_total_limit=1, # Save space
            label_smoothing_factor=0.05,
            dataloader_drop_last=False,
            # Add these for better GPU utilization
            max_grad_norm=1.0,
            logging_first_step=True,
            disable_tqdm=False,
        )

    def train(self, texts, labels):
        """Enhanced training with cross-validation and SMOTE"""
        if not self.load_model():
            return False, None
    
        try:
            # Clean and filter data
            filtered_texts = []
            filtered_labels = []
            for text, label in zip(texts, labels):
                if isinstance(text, str) and len(text.strip()) > 50:
                    processed_text = preprocess_text_for_classification(text)
                    if len(processed_text.strip()) > 50:
                        filtered_texts.append(processed_text)
                        filtered_labels.append(label)
            
            if len(filtered_texts) < 10:
                logging.error("Insufficient training data after filtering")
                return False, None
            
            logging.info(f"Filtered training data: {len(filtered_texts)} samples")
            numeric_labels = [self.label_map.get(label, 0) for label in filtered_labels]
            
            # Apply SMOTE for oversampling
            tokenized_texts = [self.tokenizer(text, max_length=config.max_length, 
                                           truncation=True, padding='max_length', 
                                           return_tensors='pt')['input_ids'].squeeze().numpy() 
                             for text in filtered_texts]
            smote = SMOTE(random_state=42)
            resampled_texts, resampled_labels = smote.fit_resample(tokenized_texts, numeric_labels)
            resampled_texts = [self.tokenizer.decode(text, skip_special_tokens=True) 
                              for text in resampled_texts]
            
            logging.info(f"After SMOTE: {len(resampled_texts)} samples")
            
            # Compute class weights
            unique_labels = np.unique(resampled_labels)
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=resampled_labels)
            label_counts = np.bincount(resampled_labels)
            imbalance_ratio = max(label_counts) / min(label_counts)
            if imbalance_ratio > 5:
                minority_class = np.argmin(label_counts)
                class_weights[minority_class] *= 2.5
            class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            
            logging.info(f"Enhanced class weights: {class_weight_dict}")
            logging.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
            logging.info(f"Label distribution: {dict(zip(*np.unique(resampled_labels, return_counts=True)))}")
            
            # Cross-validation
            kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=42)
            best_f1 = 0
            best_threshold = 0.5
            best_model_state = None
            val_results = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(resampled_texts)):
                logging.info(f"Training fold {fold + 1}/{config.n_folds}")
                train_texts = [resampled_texts[i] for i in train_idx]
                train_labels = [resampled_labels[i] for i in train_idx]
                val_texts = [resampled_texts[i] for i in val_idx]
                val_labels = [resampled_labels[i] for i in val_idx]
                
                train_dataset = CitationDataset(train_texts, train_labels, self.tokenizer)
                val_dataset = CitationDataset(val_texts, val_labels, self.tokenizer)
                
                batch_size = min(config.batch_size, len(train_texts) // 4)
                if self.device.type == 'cpu':
                    batch_size = min(batch_size, 2)
                
                training_args = self.get_training_args(batch_size)
                
                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    if torch.is_tensor(logits):
                        logits = logits.detach().cpu().numpy()
                    if torch.is_tensor(labels):
                        labels = labels.detach().cpu().numpy()
                    
                    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    
                    # Granular threshold search
                    precision, recall, thresholds = precision_recall_curve(labels, probabilities[:, 1])
                    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
                    best_idx = np.argmax(f1_scores)
                    best_f1 = f1_scores[best_idx]
                    best_threshold = thresholds[best_idx]
                    
                    final_predictions = (probabilities[:, 1] >= best_threshold).astype(int)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels, final_predictions, average='binary', pos_label=1, zero_division=0
                    )
                    accuracy = accuracy_score(labels, final_predictions)
                    
                    logging.info(f"Fold {fold + 1} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                    logging.info(f"Fold {fold + 1} - Best threshold: {best_threshold:.3f}")
                    
                    return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy, 
                           'best_threshold': best_threshold}
                
                trainer = WeightedTrainer(
                    class_weights=class_weight_dict,
                    alpha=0.25,
                    gamma=3.0,
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=compute_metrics,
                    tokenizer=self.tokenizer,
                )
                
                trainer.train()
                eval_results = trainer.evaluate()
                
                if eval_results['eval_f1'] > best_f1:
                    best_f1 = eval_results['eval_f1']
                    best_threshold = eval_results['eval_best_threshold']
                    best_model_state = self.model.state_dict()
                
                val_predictions = [self.classify_with_threshold(text, eval_results['eval_best_threshold']) 
                                 for text in val_texts]
                val_results.append({
                    'texts': val_texts,
                    'true_labels': [self.reverse_label_map[label] for label in val_labels],
                    'predictions': val_predictions,
                    'metrics': eval_results
                })
            
            # Load best model
            if best_model_state:
                self.model.load_state_dict(best_model_state)
                self.best_threshold = best_threshold
                self.is_trained = True
            
            logging.info(f"Best F1 across folds: {best_f1:.4f}")
            logging.info(f"Best threshold: {best_threshold:.3f}")
            return True, val_results[-1]
        
        except Exception as e:
            logging.error(f"Training error: {e}")
            return False, None
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def classify_with_threshold(self, text, threshold=0.5):
        """Enhanced classification with GPU verification"""
        if self.is_trained and self.model is not None and self.tokenizer is not None:
            try:
                self.model.eval()
                processed_text = preprocess_text_for_classification(text)
                
                if len(processed_text.strip()) < 50:
                    processed_text = text[:2000] if len(text) > 2000 else text
                    
                inputs = self.tokenizer(
                    processed_text,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=config.max_length
                )
                
                # CRITICAL: Move inputs to same device as model with non_blocking
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)
                    primary_prob = probabilities[0, 1].item()
                    prediction = 1 if primary_prob >= threshold else 0
                    
                return self.reverse_label_map[prediction]
                
            except Exception as e:
                logging.error(f"Classification failed: {e}")
                return self._enhanced_fallback_classify(text)
        else:
            return self._enhanced_fallback_classify(text)

    def _enhanced_fallback_classify(self, text):
        """Enhanced heuristic classification"""
        if not isinstance(text, str):
            text = str(text)
        text_lower = text.lower()
        strong_primary = [
            'data availability statement', 'supplementary data', 'supporting information',
            'raw data', 'source data', 'original data', 'deposited at', 'archived at',
            'figshare', 'dryad', 'zenodo', 'mendeley data', 'github repository'
        ]
        moderate_primary = [
            'additional file', 'dataset', 'data repository', 'uploaded', 'available at',
            'data are available', 'supporting material'
        ]
        strong_secondary = [
            'downloaded from', 'obtained from', 'retrieved from', 'accessed from',
            'geo database', 'ncbi', 'arrayexpress', 'ensembl', 'uniprot',
            'publicly available database', 'previously published'
        ]
        moderate_secondary = [
            'public repository', 'database query', 'reference genome',
            'annotation', 'existing dataset'
        ]
        primary_score = (
            sum(3 for indicator in strong_primary if indicator in text_lower) +
            sum(1 for indicator in moderate_primary if indicator in text_lower)
        )
        secondary_score = (
            sum(3 for indicator in strong_secondary if indicator in text_lower) +
            sum(1 for indicator in moderate_secondary if indicator in text_lower)
        )
        return 'Primary' if primary_score >= secondary_score else 'Secondary'

def process_training_data():
    """Process training data and train classifier"""
    processor = DocumentProcessor(config.use_pdf)
    detector = EnhancedDatasetDetector()
    classifier = ScientificClassifier()

    if not os.path.exists(config.train_labels_path):
        logging.error(f"Training labels file not found: {config.train_labels_path}")
        return detector, classifier, None

    try:
        train_labels = pd.read_csv(config.train_labels_path)
        # Clean duplicates and normalize dataset_id
        train_labels['dataset_id'] = train_labels['dataset_id'].str.replace(r'\.$', '', regex=True)
        train_labels = train_labels.drop_duplicates(subset=['article_id', 'dataset_id', 'type'])
        logging.info(f"Loaded {len(train_labels)} training labels after cleaning")
        logging.info(f"Label distribution:\n{train_labels['type'].value_counts()}")
    except Exception as e:
        logging.error(f"Failed to read train_labels.csv: {e}")
        return detector, classifier, None

    texts = []
    labels = []
    processed_count = 0
    
    for _, row in tqdm(train_labels.iterrows(), total=len(train_labels), desc="Processing training data"):
        article_id = row['article_id']
        doc_path = processor.get_document_path(article_id, is_train=True)
        if os.path.exists(doc_path):
            text = processor.process_document(doc_path)
            if text and text.strip():
                texts.append(text)
                labels.append(row['type'])
                processed_count += 1
        if processed_count % 50 == 0:
            gc.collect()

    logging.info(f"Processed {len(texts)} training documents")
    
    if not texts:
        logging.error("No training texts processed")
        return detector, classifier, None

    success, validation_results = classifier.train(texts, labels)
    
    if success and validation_results:
        logging.info("✓ Model training completed successfully")
        val_df = pd.DataFrame({
            'true_label': validation_results['true_labels'],
            'predicted_label': validation_results['predictions'],
            'text_sample': [text[:200] + '...' if len(text) > 200 else text 
                           for text in validation_results['texts']]
        })
        try:
            val_df.to_csv(config.validation_output_path, index=False)
            logging.info(f"Validation results saved to {config.validation_output_path}")
        except Exception as e:
            logging.error(f"Failed exaggerate Failed to save validation results: {e}")
        report = classification_report(
            validation_results['true_labels'], 
            validation_results['predictions']
        )
        logging.info(f"Classification Report:\n{report}")
    else:
        logging.warning("Model training failed, using fallback classification")

    return detector, classifier, validation_results

def process_test_data(detector, classifier):
    """Process test data and generate predictions"""
    processor = DocumentProcessor(config.use_pdf)
    submission_data = []

    test_dir = config.test_pdf_path if config.use_pdf else config.test_xml_path
    extension = '.pdf' if config.use_pdf else '.xml'
    test_files = []
    if os.path.exists(test_dir):
        for root, _, files in os.walk(test_dir):
            for f in files:
                if f.lower().endswith(extension):
                    test_files.append(os.path.join(root, f))

    logging.info(f"Processing {len(test_files)} test files ({processor.doc_type})")

    processed_count = 0
    for file_path in tqdm(test_files, desc="Processing test data"):
        try:
            article_id = os.path.splitext(os.path.basename(file_path))[0]
            text = processor.process_document(file_path)
            if text and text.strip():
                datasets = detector.detect_datasets(text)
                if datasets:
                    threshold = getattr(classifier, 'best_threshold', 0.5)
                    classification = classifier.classify_with_threshold(text, threshold)
                    for dataset_id in datasets:
                        submission_data.append({
                            'article_id': article_id,
                            'dataset_id': dataset_id,
                            'type': classification
                        })
                    processed_count += 1
            if processed_count % 20 == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    submission_df = pd.DataFrame(submission_data, columns=['article_id', 'dataset_id', 'type'])
    if not submission_df.empty:
        submission_df.reset_index(drop=True, inplace=True)
        submission_df.insert(0, 'row_id', range(len(submission_df)))
    else:
        submission_df = pd.DataFrame(columns=['row_id', 'article_id', 'dataset_id', 'type'])
    
    if not submission_df.empty:
        logging.info(f"Generated {len(submission_df)} predictions")
        logging.info(f"Prediction distribution:\n{submission_df['type'].value_counts()}")
        logging.info(f"Unique articles: {submission_df['article_id'].nunique()}")
        logging.info(f"Unique datasets: {submission_df['dataset_id'].nunique()}")
        logging.info(f"Row IDs range from 0 to {len(submission_df)-1}")
    else:
        logging.warning("No predictions generated")

    try:
        submission_df.to_csv(config.output_path, index=False)
        logging.info(f"✓ Submission saved: {config.output_path}")
        logging.info(f"✓ Submission format: {list(submission_df.columns)}")
        if not submission_df.empty:
            logging.info(f"First 3 rows of submission:\n{submission_df.head(3)}")
    except Exception as e:
        logging.error(f"Failed to save submission: {e}")

    return submission_df

def main():
    """Main execution function with enhanced GPU monitoring"""
    try:
        logging.info("=" * 60)
        logging.info("🚀 ENHANCED DUAL GPU PIPELINE")
        logging.info("=" * 60)
        
        # Clear memory at start
        clear_gpu_memory()
        
        # Setup device (will now use both GPUs)
        device = setup_device()
        
        # Show GPU info
        if device.type == 'cuda':
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logging.info(f"🔧 GPU {i}: {props.name} - {props.total_memory/1024**3:.1f} GB")
        
        # Initialize monitor
        monitor = GPUMonitor(device)
        monitor.log_memory_usage("Pipeline Start")
        
        # Rest of your main function...
        logging.info(f"🤖 ACTIVE MODEL: {BEST_MODEL}")
        
        # Training phase
        start_time = time.time()
        detector, classifier, validation_results = process_training_data()
        training_time = time.time() - start_time
        
        monitor.log_memory_usage("After Training")
        
        # Inference phase
        start_time = time.time()
        submission_df = process_test_data(detector, classifier)
        inference_time = time.time() - start_time
        
        monitor.log_memory_usage("After Inference")
        
        logging.info(f"✅ TRAINING: {training_time:.2f}s, INFERENCE: {inference_time:.2f}s")
        
    except Exception as e:
        logging.error(f"💥 PIPELINE ERROR: {e}")
    finally:
        clear_gpu_memory()
        
        # Final GPU usage report
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
                logging.info(f"📊 GPU {i} Max Usage: {max_allocated:.2f} GB")
                if max_allocated > 0.5:
                    logging.info(f"✅ GPU {i} WAS ACTIVELY USED!")
                else:
                    logging.warning(f"⚠️ GPU {i} had low usage")

if __name__ == "__main__":
    main()