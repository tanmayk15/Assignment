"""
QA Pair Generator
Automatically generates question-answer pairs from PCB images and bounding boxes
"""

import json
import random
from typing import Dict, List, Tuple
import numpy as np


class QAGenerator:
    """Generate diverse QA pairs for PCB defect detection"""
    
    def __init__(self):
        self.defect_types = [
            'solder_bridge', 'cold_joint', 'tombstone', 'insufficient_solder',
            'excessive_solder', 'missing_component', 'wrong_component',
            'damaged_component', 'misaligned_component', 'lifted_pad'
        ]
        
        self.spatial_relations = [
            'above', 'below', 'left of', 'right of', 'near', 
            'far from', 'adjacent to', 'diagonal to'
        ]
        
        self.question_templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different query types"""
        return {
            'counting': [
                "How many {defect_type} defects are there?",
                "Count the {defect_type}s",
                "What is the number of {defect_type} defects?",
                "How many {defect_type} issues can you see?",
                "Tell me the count of {defect_type} defects"
            ],
            'existence': [
                "Are there any {defect_type} defects?",
                "Is there a {defect_type} present?",
                "Do you see any {defect_type}?",
                "Can you find any {defect_type} defects?",
                "Does this PCB have {defect_type}?"
            ],
            'localization': [
                "Where is the {defect_type}?",
                "Show me the location of {defect_type}",
                "Point to the {defect_type} defect",
                "What are the coordinates of the {defect_type}?",
                "Locate the {defect_type} defects"
            ],
            'spatial': [
                "What is {relation} the {defect_type}?",
                "What defects are {relation} the {target}?",
                "Describe the position of {defect_type} relative to {target}",
                "Is {defect_type1} {relation} {defect_type2}?"
            ],
            'total_counting': [
                "How many total defects are there?",
                "Count all defects",
                "What is the total number of issues?",
                "How many defects in total?"
            ],
            'severity': [
                "What is the severity of the {defect_type}?",
                "Is the {defect_type} critical?",
                "How severe is this defect?"
            ]
        }
    
    def generate_qa_pairs(
        self,
        image_id: str,
        bboxes: List[List[float]],
        defect_labels: List[str],
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Generate comprehensive QA pairs for an image
        
        Args:
            image_id: Unique image identifier
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            defect_labels: List of defect type labels
            metadata: Optional metadata (image size, etc.)
        
        Returns:
            List of QA pair dictionaries
        """
        qa_pairs = []
        
        # 1. Counting questions (per defect type)
        qa_pairs.extend(self._generate_counting_questions(defect_labels))
        
        # 2. Existence questions
        qa_pairs.extend(self._generate_existence_questions(defect_labels))
        
        # 3. Localization questions
        qa_pairs.extend(self._generate_localization_questions(bboxes, defect_labels))
        
        # 4. Spatial relationship questions
        if len(bboxes) >= 2:
            qa_pairs.extend(self._generate_spatial_questions(bboxes, defect_labels))
        
        # 5. Total counting question
        qa_pairs.extend(self._generate_total_counting_questions(defect_labels))
        
        # 6. Negative samples (hallucination prevention)
        qa_pairs.extend(self._generate_negative_samples(defect_labels))
        
        # Add image_id to all pairs
        for pair in qa_pairs:
            pair['image_id'] = image_id
            pair['num_defects'] = len(bboxes)
        
        return qa_pairs
    
    def _generate_counting_questions(self, defect_labels: List[str]) -> List[Dict]:
        """Generate counting questions for each defect type"""
        qa_pairs = []
        
        # Count each defect type
        defect_counts = {}
        for label in defect_labels:
            defect_counts[label] = defect_counts.get(label, 0) + 1
        
        for defect_type, count in defect_counts.items():
            template = random.choice(self.question_templates['counting'])
            question = template.format(defect_type=defect_type)
            
            # Multiple answer formats
            answers = [
                str(count),
                f"There are {count} {defect_type} defects.",
                f"{count} {defect_type}{'s' if count != 1 else ''}"
            ]
            
            qa_pairs.append({
                'question': question,
                'answer': random.choice(answers),
                'structured_answer': {
                    'count': count,
                    'defect_type': defect_type,
                    'query_type': 'counting'
                },
                'type': 'counting'
            })
        
        return qa_pairs
    
    def _generate_existence_questions(self, defect_labels: List[str]) -> List[Dict]:
        """Generate yes/no existence questions"""
        qa_pairs = []
        present_defects = set(defect_labels)
        
        # Questions about present defects
        for defect_type in random.sample(list(present_defects), min(3, len(present_defects))):
            template = random.choice(self.question_templates['existence'])
            question = template.format(defect_type=defect_type)
            
            answers = ["Yes", f"Yes, there are {defect_labels.count(defect_type)} {defect_type} defects."]
            
            qa_pairs.append({
                'question': question,
                'answer': random.choice(answers),
                'structured_answer': {
                    'exists': True,
                    'defect_type': defect_type,
                    'count': defect_labels.count(defect_type),
                    'query_type': 'existence'
                },
                'type': 'existence'
            })
        
        # Questions about absent defects (negative samples)
        absent_defects = set(self.defect_types) - present_defects
        for defect_type in random.sample(list(absent_defects), min(2, len(absent_defects))):
            template = random.choice(self.question_templates['existence'])
            question = template.format(defect_type=defect_type)
            
            qa_pairs.append({
                'question': question,
                'answer': "No",
                'structured_answer': {
                    'exists': False,
                    'defect_type': defect_type,
                    'query_type': 'existence'
                },
                'type': 'existence'
            })
        
        return qa_pairs
    
    def _generate_localization_questions(
        self,
        bboxes: List[List[float]],
        defect_labels: List[str]
    ) -> List[Dict]:
        """Generate localization questions with bounding boxes"""
        qa_pairs = []
        
        # Select a few defects to localize
        indices = random.sample(range(len(bboxes)), min(3, len(bboxes)))
        
        for idx in indices:
            bbox = bboxes[idx]
            label = defect_labels[idx]
            
            template = random.choice(self.question_templates['localization'])
            question = template.format(defect_type=label)
            
            # Format bbox
            bbox_str = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
            
            answers = [
                f"The {label} is at coordinates {bbox_str}",
                json.dumps({'bbox': [int(b) for b in bbox], 'confidence': 1.0}),
                f"Located at {bbox_str}"
            ]
            
            qa_pairs.append({
                'question': question,
                'answer': random.choice(answers),
                'structured_answer': {
                    'bbox': [int(b) for b in bbox],
                    'defect_type': label,
                    'confidence': 1.0,
                    'query_type': 'localization'
                },
                'type': 'localization'
            })
        
        return qa_pairs
    
    def _generate_spatial_questions(
        self,
        bboxes: List[List[float]],
        defect_labels: List[str]
    ) -> List[Dict]:
        """Generate spatial relationship questions"""
        qa_pairs = []
        
        # Generate 2-3 spatial questions
        for _ in range(min(2, len(bboxes) - 1)):
            idx1, idx2 = random.sample(range(len(bboxes)), 2)
            
            bbox1 = bboxes[idx1]
            bbox2 = bboxes[idx2]
            label1 = defect_labels[idx1]
            label2 = defect_labels[idx2]
            
            # Compute spatial relationship
            relation = self._compute_spatial_relation(bbox1, bbox2)
            
            # Generate question
            questions = [
                f"What is {relation} the {label1}?",
                f"What defects are {relation} the {label1}?",
                f"Is the {label2} {relation} the {label1}?"
            ]
            
            question = random.choice(questions)
            
            if "What" in question:
                answer = f"The {label2} is {relation} the {label1}"
            else:
                answer = "Yes"
            
            qa_pairs.append({
                'question': question,
                'answer': answer,
                'structured_answer': {
                    'object1': {'bbox': [int(b) for b in bbox1], 'type': label1},
                    'object2': {'bbox': [int(b) for b in bbox2], 'type': label2},
                    'relation': relation,
                    'query_type': 'spatial'
                },
                'type': 'spatial'
            })
        
        return qa_pairs
    
    def _generate_total_counting_questions(self, defect_labels: List[str]) -> List[Dict]:
        """Generate questions about total defect count"""
        total_count = len(defect_labels)
        
        template = random.choice(self.question_templates['total_counting'])
        
        answers = [
            str(total_count),
            f"There are {total_count} defects in total.",
            f"{total_count} total defects"
        ]
        
        return [{
            'question': template,
            'answer': random.choice(answers),
            'structured_answer': {
                'total_count': total_count,
                'query_type': 'total_counting'
            },
            'type': 'total_counting'
        }]
    
    def _generate_negative_samples(self, defect_labels: List[str]) -> List[Dict]:
        """Generate negative samples to prevent hallucination"""
        qa_pairs = []
        present_defects = set(defect_labels)
        absent_defects = list(set(self.defect_types) - present_defects)
        
        # Ask about non-existent defects
        for defect_type in random.sample(absent_defects, min(2, len(absent_defects))):
            qa_pairs.append({
                'question': f"How many {defect_type} defects are there?",
                'answer': "0",
                'structured_answer': {
                    'count': 0,
                    'defect_type': defect_type,
                    'query_type': 'counting'
                },
                'type': 'negative_counting',
                'is_negative': True
            })
        
        return qa_pairs
    
    def _compute_spatial_relation(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> str:
        """Compute spatial relationship between two bounding boxes"""
        # Centers
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        
        # Differences
        dx = cx2 - cx1
        dy = cy2 - cy1
        
        # Distance
        distance = np.sqrt(dx**2 + dy**2)
        
        # Determine relation
        if abs(dx) > abs(dy):
            if dx > 0:
                relation = "right of"
            else:
                relation = "left of"
        else:
            if dy > 0:
                relation = "below"
            else:
                relation = "above"
        
        # Modify based on distance
        if distance < 100:
            relation = "near"
        
        return relation
    
    def generate_dataset(
        self,
        annotations: List[Dict],
        output_path: str = 'qa_dataset.json'
    ):
        """
        Generate complete QA dataset from annotations
        
        Args:
            annotations: List of image annotations with bboxes and labels
            output_path: Path to save generated dataset
        """
        print(f"Generating QA pairs from {len(annotations)} images...")
        
        all_qa_pairs = []
        stats = {'counting': 0, 'existence': 0, 'localization': 0, 'spatial': 0, 'total': 0}
        
        for i, annotation in enumerate(annotations):
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(annotations)} images...")
            
            pairs = self.generate_qa_pairs(
                image_id=annotation['image_id'],
                bboxes=annotation['bboxes'],
                defect_labels=annotation['labels']
            )
            
            all_qa_pairs.extend(pairs)
            
            for pair in pairs:
                pair_type = pair['type'].split('_')[0]
                if pair_type in stats:
                    stats[pair_type] += 1
        
        # Save dataset
        with open(output_path, 'w') as f:
            json.dump(all_qa_pairs, f, indent=2)
        
        print(f"\n✓ Generated {len(all_qa_pairs)} QA pairs")
        print(f"  Counting: {stats['counting']}")
        print(f"  Existence: {stats['existence']}")
        print(f"  Localization: {stats['localization']}")
        print(f"  Spatial: {stats['spatial']}")
        print(f"  Saved to {output_path}")
        
        return all_qa_pairs


def main():
    """Test QA generator"""
    print("=" * 70)
    print("QA PAIR GENERATION DEMO")
    print("=" * 70)
    
    generator = QAGenerator()
    
    # Example annotation
    example_annotation = {
        'image_id': 'pcb_00001',
        'bboxes': [
            [120, 340, 145, 365],
            [200, 150, 225, 175],
            [450, 280, 475, 305],
            [350, 420, 375, 445]
        ],
        'labels': ['solder_bridge', 'solder_bridge', 'cold_joint', 'tombstone']
    }
    
    # Generate QA pairs
    qa_pairs = generator.generate_qa_pairs(
        image_id=example_annotation['image_id'],
        bboxes=example_annotation['bboxes'],
        defect_labels=example_annotation['labels']
    )
    
    print(f"\n✓ Generated {len(qa_pairs)} QA pairs\n")
    
    # Display examples
    print("EXAMPLES:")
    print("-" * 70)
    
    for i, pair in enumerate(qa_pairs[:10], 1):
        print(f"\n{i}. Type: {pair['type']}")
        print(f"   Q: {pair['question']}")
        print(f"   A: {pair['answer']}")
        if 'bbox' in pair.get('structured_answer', {}):
            print(f"   BBox: {pair['structured_answer']['bbox']}")
    
    print("\n" + "=" * 70)
    print(f"Total QA pairs generated: {len(qa_pairs)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
