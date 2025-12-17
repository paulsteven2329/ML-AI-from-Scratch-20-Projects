"""
Project 14: LLM-Based AI Resume Reviewer
"Using LLMs to solve real problems"

This project demonstrates how to build production-ready applications using
Large Language Models (LLMs). You'll create an intelligent resume reviewer
that provides detailed feedback, scores candidates, and generates insights.

Learning Objectives:
1. Master prompt engineering techniques for LLMs
2. Build robust LLM applications with error handling
3. Implement structured output parsing and validation
4. Create business intelligence from unstructured text
5. Design scalable LLM-powered services

Business Context:
LLM applications are transforming recruitment:
- Resume screening: 75% time reduction vs manual review
- Bias reduction: Standardized, objective evaluation criteria
- Scalability: Process thousands of resumes simultaneously
- Consistency: Same evaluation standards across all candidates
- Cost efficiency: $50/hr recruiter vs $0.10/analysis
"""

import os
import json
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Integration libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Install with: pip install openai")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers library not available. Install with: pip install transformers torch")

@dataclass
class ResumeAnalysis:
    """Structured output for resume analysis"""
    candidate_name: str
    overall_score: float  # 0-100
    experience_score: float
    education_score: float
    skills_score: float
    communication_score: float
    
    strengths: List[str]
    weaknesses: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    
    years_of_experience: int
    education_level: str
    key_skills: List[str]
    previous_companies: List[str]
    
    red_flags: List[str]
    interview_questions: List[str]
    salary_estimate: str
    fit_assessment: str
    
    confidence_score: float  # Model's confidence in the analysis
    processing_time: float

class PromptEngineer:
    """
    Advanced prompt engineering for LLM applications
    
    Prompt engineering is crucial for reliable LLM performance:
    - Clear instructions reduce ambiguity
    - Examples improve output quality (few-shot learning)
    - Structured formats enable parsing
    - Context management prevents hallucination
    """
    
    @staticmethod
    def create_resume_analysis_prompt(resume_text: str, job_description: str = None) -> str:
        """
        Create a comprehensive prompt for resume analysis
        
        Key techniques used:
        1. Role definition: Tell the LLM what it is
        2. Task specification: Clear, specific instructions
        3. Output format: Structured JSON for easy parsing
        4. Examples: Show desired output format
        5. Constraints: Guard against common errors
        """
        
        base_prompt = """
You are an expert HR professional and resume reviewer with 15+ years of experience in talent acquisition. 
Your task is to analyze resumes thoroughly and provide actionable insights for hiring decisions.

ANALYSIS FRAMEWORK:
1. Overall Assessment (0-100 score)
2. Detailed Scoring:
   - Experience Relevance (0-100)
   - Education Quality (0-100) 
   - Skills Match (0-100)
   - Communication Quality (0-100)

3. Qualitative Analysis:
   - Top 3-5 Strengths
   - Top 3-5 Areas for Improvement
   - Missing Critical Skills
   - Red Flags (if any)

4. Business Intelligence:
   - Years of Experience (estimate)
   - Education Level
   - Key Technical/Soft Skills
   - Previous Companies
   - Salary Estimate Range
   - Interview Questions (5 behavioral + 5 technical)

5. Recommendation:
   - Fit Assessment (Excellent/Good/Fair/Poor)
   - Specific Recommendations for candidate

OUTPUT FORMAT: Provide analysis in valid JSON format with this structure:
{
    "candidate_name": "Extract from resume or 'Not Provided'",
    "overall_score": 85.5,
    "experience_score": 90.0,
    "education_score": 80.0,
    "skills_score": 88.0,
    "communication_score": 85.0,
    "strengths": ["Strength 1", "Strength 2", "Strength 3"],
    "weaknesses": ["Weakness 1", "Weakness 2"],
    "missing_skills": ["Skill 1", "Skill 2"],
    "recommendations": ["Recommendation 1", "Recommendation 2"],
    "years_of_experience": 5,
    "education_level": "Bachelor's Degree",
    "key_skills": ["Python", "Machine Learning", "Leadership"],
    "previous_companies": ["Company A", "Company B"],
    "red_flags": ["Flag 1 if any"],
    "interview_questions": ["Question 1", "Question 2", "Question 3"],
    "salary_estimate": "$80,000 - $100,000",
    "fit_assessment": "Good",
    "confidence_score": 0.85
}

IMPORTANT GUIDELINES:
- Be objective and data-driven in your assessment
- Consider both technical and soft skills
- Flag any inconsistencies or red flags
- Provide actionable, specific recommendations
- Estimate salary based on experience, skills, and location context
- Generate thoughtful interview questions that probe key areas
"""
        
        job_context = ""
        if job_description:
            job_context = f"""
JOB REQUIREMENTS CONTEXT:
{job_description}

Additional Instructions:
- Score skills match specifically against job requirements
- Highlight relevant experience for this role
- Identify gaps between candidate and job requirements
- Tailor interview questions to this specific position
"""
        
        resume_section = f"""
RESUME TO ANALYZE:
{resume_text}

Please provide your analysis following the exact JSON format specified above.
"""
        
        return base_prompt + job_context + resume_section
    
    @staticmethod
    def create_batch_comparison_prompt(resumes: List[Dict], job_description: str) -> str:
        """Create prompt for comparing multiple candidates"""
        
        prompt = f"""
You are an expert HR professional tasked with ranking candidates for a specific position.

JOB DESCRIPTION:
{job_description}

CANDIDATES TO COMPARE:
"""
        
        for i, resume_data in enumerate(resumes, 1):
            prompt += f"\nCANDIDATE {i}:\n{resume_data['text']}\n"
        
        prompt += """
TASK:
1. Rank candidates from best to worst fit for this position
2. Provide scoring rationale for each candidate
3. Identify top 3 candidates for interview
4. Highlight unique strengths of each candidate

OUTPUT FORMAT (JSON):
{
    "ranking": [
        {
            "rank": 1,
            "candidate_name": "Name or 'Candidate 1'",
            "overall_score": 92.0,
            "key_strengths": ["Strength 1", "Strength 2"],
            "rationale": "Why this candidate ranks here"
        }
    ],
    "top_3_for_interview": ["Candidate names"],
    "hiring_recommendation": "Overall recommendation for hiring process"
}
"""
        return prompt

class LLMClient:
    """
    Abstracted LLM client supporting multiple providers
    
    This design allows switching between OpenAI, HuggingFace, or local models
    without changing application logic.
    """
    
    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "huggingface":
            self._init_huggingface()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available")
        
        # Use environment variable or provided API key
        if self.api_key:
            openai.api_key = self.api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.use_mock = True
            return
        
        self.model = self.model or "gpt-3.5-turbo"
        self.use_mock = False
        logger.info(f"OpenAI client initialized with model: {self.model}")
    
    def _init_huggingface(self):
        """Initialize HuggingFace client"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
        
        self.model = self.model or "microsoft/DialoGPT-medium"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.model)
            self.use_mock = False
            logger.info(f"HuggingFace model loaded: {self.model}")
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace model: {e}")
            self.use_mock = True
    
    def generate_response(self, prompt: str, max_tokens: int = 2000, 
                         temperature: float = 0.3) -> str:
        """
        Generate response from LLM
        
        Parameters:
        - prompt: Input text for the model
        - max_tokens: Maximum response length
        - temperature: Creativity level (0.0 = deterministic, 1.0 = creative)
        """
        
        if self.use_mock:
            return self._mock_response(prompt)
        
        try:
            if self.provider == "openai":
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert HR professional."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            
            elif self.provider == "huggingface":
                inputs = self.tokenizer.encode(prompt, return_tensors='pt')
                
                with torch.no_grad():
                    outputs = self.llm_model.generate(
                        inputs, 
                        max_length=min(inputs.shape[1] + max_tokens, 4096),
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the original prompt from response
                response = response[len(prompt):].strip()
                return response
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing when LLM is unavailable"""
        return """{
    "candidate_name": "John Smith",
    "overall_score": 85.5,
    "experience_score": 88.0,
    "education_score": 82.0,
    "skills_score": 90.0,
    "communication_score": 83.0,
    "strengths": [
        "Strong technical background in Python and machine learning",
        "Progressive career growth with increasing responsibilities", 
        "Good educational foundation with relevant degree",
        "Experience with modern development tools and practices"
    ],
    "weaknesses": [
        "Limited experience with cloud platforms",
        "Could benefit from more leadership experience",
        "Resume formatting could be improved"
    ],
    "missing_skills": ["AWS", "Docker", "Team Leadership"],
    "recommendations": [
        "Consider pursuing cloud certifications to strengthen profile",
        "Seek opportunities for team leadership roles",
        "Improve resume formatting and structure"
    ],
    "years_of_experience": 5,
    "education_level": "Bachelor's Degree",
    "key_skills": ["Python", "Machine Learning", "SQL", "Git"],
    "previous_companies": ["Tech Corp", "Data Solutions Inc"],
    "red_flags": [],
    "interview_questions": [
        "Tell me about a challenging ML project you've worked on",
        "How do you approach debugging complex data issues?",
        "Describe a time when you had to learn a new technology quickly"
    ],
    "salary_estimate": "$75,000 - $95,000",
    "fit_assessment": "Good",
    "confidence_score": 0.75
}"""

class ResumeReviewer:
    """
    AI-powered resume review system with comprehensive analysis
    """
    
    def __init__(self, llm_provider: str = "openai", model: str = None, 
                 output_dir: str = "outputs"):
        self.llm_client = LLMClient(provider=llm_provider, model=model)
        self.output_dir = output_dir
        self.analysis_history = []
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/individual_analyses", exist_ok=True)
        os.makedirs(f"{output_dir}/batch_analyses", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    
    def create_sample_resumes(self) -> List[Dict[str, str]]:
        """
        Create sample resumes for demonstration
        
        In production, these would come from:
        - PDF parsing (using PyPDF2, pdfplumber)
        - ATS integrations
        - Direct text input from hiring platforms
        """
        
        sample_resumes = [
            {
                "name": "John Smith - Senior Data Scientist",
                "text": """
John Smith
Senior Data Scientist
Email: john.smith@email.com | Phone: (555) 123-4567

EXPERIENCE:
Senior Data Scientist - TechCorp (2020-Present)
• Led machine learning initiatives for customer segmentation, improving conversion rates by 25%
• Built end-to-end ML pipelines using Python, Spark, and AWS
• Mentored 3 junior data scientists and established ML best practices
• Deployed models serving 1M+ predictions daily with 99.9% uptime

Data Scientist - Analytics Solutions (2018-2020)
• Developed predictive models for inventory optimization, reducing waste by 15%
• Created automated reporting dashboards using Tableau and Python
• Collaborated with product teams to define KPIs and success metrics
• Presented insights to C-level executives quarterly

Junior Data Analyst - StartupXYZ (2016-2018)
• Analyzed user behavior data to identify growth opportunities
• Built SQL queries and data visualizations for stakeholder reporting
• Assisted in A/B testing framework implementation

EDUCATION:
M.S. Computer Science - Stanford University (2016)
B.S. Mathematics - UC Berkeley (2014)

SKILLS:
Programming: Python, R, SQL, Scala
ML/AI: Scikit-learn, TensorFlow, PyTorch, MLflow
Cloud: AWS (EC2, S3, SageMaker), Docker, Kubernetes
Visualization: Tableau, Matplotlib, Seaborn
Databases: PostgreSQL, MongoDB, Redis
"""
            },
            {
                "name": "Sarah Johnson - ML Engineer",
                "text": """
Sarah Johnson
Machine Learning Engineer
sarah.johnson@gmail.com | (555) 987-6543

PROFESSIONAL EXPERIENCE:

ML Engineer - AI Innovations Inc (2019-Present)
• Designed and implemented real-time recommendation systems serving 10M+ users
• Optimized model inference latency by 40% through efficient feature engineering
• Built MLOps pipelines for automated model training, validation, and deployment
• Led cross-functional projects with engineering, product, and business teams

Software Engineer - DataTech Solutions (2017-2019)
• Developed scalable data processing systems using Apache Spark and Kafka
• Implemented monitoring and alerting systems for production ML models
• Collaborated on microservices architecture supporting ML applications

Intern - Research Lab (Summer 2016)
• Conducted research on natural language processing and deep learning
• Published paper on attention mechanisms in neural machine translation
• Developed prototypes using TensorFlow and contributed to open-source projects

EDUCATION:
M.S. Artificial Intelligence - MIT (2017)
Thesis: "Attention Mechanisms in Sequential Learning"
B.S. Computer Science - Carnegie Mellon (2015)
GPA: 3.8/4.0

TECHNICAL SKILLS:
Languages: Python, Java, C++, Go
Frameworks: TensorFlow, PyTorch, Keras, Scikit-learn
Infrastructure: Kubernetes, Docker, Terraform
Databases: PostgreSQL, Cassandra, Elasticsearch
Tools: Git, Jenkins, Grafana, Jupyter

PUBLICATIONS:
• "Enhanced Attention for Neural Machine Translation" - ICML 2017
• "Scalable Real-time Recommendations" - KDD 2020

CERTIFICATIONS:
• AWS Certified ML Specialty (2021)
• Google Cloud Professional ML Engineer (2020)
"""
            },
            {
                "name": "Mike Chen - Junior Developer",
                "text": """
Mike Chen
Software Developer
mikechen.dev@email.com | (555) 246-8135

WORK EXPERIENCE:

Junior Software Developer - WebDev Co (2022-Present)
• Developing web applications using React, Node.js, and MongoDB
• Working on bug fixes and feature implementations
• Participating in code reviews and agile development process
• Learning about database optimization and API design

Intern - Local Startup (Summer 2021)
• Helped build mobile app features using React Native
• Fixed bugs and improved user interface components
• Learned about version control with Git and collaborative development

EDUCATION:
B.S. Computer Science - State University (2022)
Relevant Coursework: Data Structures, Algorithms, Database Systems, Software Engineering
Senior Project: E-commerce platform with recommendation system

TECHNICAL SKILLS:
Programming: JavaScript, Python, Java
Frontend: React, HTML, CSS, Bootstrap
Backend: Node.js, Express.js
Databases: MongoDB, MySQL
Tools: Git, VS Code, npm

PROJECTS:
• Personal Portfolio Website - Built with React and deployed on Netlify
• Todo App - Full-stack application using MERN stack
• Weather App - Frontend application consuming REST APIs

CERTIFICATIONS:
• freeCodeCamp Responsive Web Design (2021)
• Coursera JavaScript Specialization (2021)
"""
            }
        ]
        
        return sample_resumes
    
    def analyze_resume(self, resume_text: str, job_description: str = None, 
                      candidate_name: str = None) -> ResumeAnalysis:
        """
        Comprehensive resume analysis using LLM
        
        Returns structured analysis with scores, insights, and recommendations
        """
        start_time = time.time()
        
        # Create analysis prompt
        prompt = PromptEngineer.create_resume_analysis_prompt(resume_text, job_description)
        
        # Generate LLM response
        logger.info("Generating LLM analysis...")
        response = self.llm_client.generate_response(prompt, max_tokens=2000, temperature=0.3)
        
        # Parse JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
            
            # Create ResumeAnalysis object
            analysis = ResumeAnalysis(
                candidate_name=response_json.get('candidate_name', candidate_name or 'Unknown'),
                overall_score=float(response_json.get('overall_score', 0)),
                experience_score=float(response_json.get('experience_score', 0)),
                education_score=float(response_json.get('education_score', 0)),
                skills_score=float(response_json.get('skills_score', 0)),
                communication_score=float(response_json.get('communication_score', 0)),
                strengths=response_json.get('strengths', []),
                weaknesses=response_json.get('weaknesses', []),
                missing_skills=response_json.get('missing_skills', []),
                recommendations=response_json.get('recommendations', []),
                years_of_experience=int(response_json.get('years_of_experience', 0)),
                education_level=response_json.get('education_level', 'Not specified'),
                key_skills=response_json.get('key_skills', []),
                previous_companies=response_json.get('previous_companies', []),
                red_flags=response_json.get('red_flags', []),
                interview_questions=response_json.get('interview_questions', []),
                salary_estimate=response_json.get('salary_estimate', 'Not estimated'),
                fit_assessment=response_json.get('fit_assessment', 'Unknown'),
                confidence_score=float(response_json.get('confidence_score', 0.5)),
                processing_time=time.time() - start_time
            )
            
            # Store analysis
            self.analysis_history.append(analysis)
            
            return analysis
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response was: {response[:500]}...")
            
            # Return default analysis with error indication
            return ResumeAnalysis(
                candidate_name=candidate_name or "Unknown",
                overall_score=50.0,
                experience_score=50.0,
                education_score=50.0,
                skills_score=50.0,
                communication_score=50.0,
                strengths=["Analysis parsing failed"],
                weaknesses=["Unable to complete analysis"],
                missing_skills=[],
                recommendations=["Please review resume manually"],
                years_of_experience=0,
                education_level="Unknown",
                key_skills=[],
                previous_companies=[],
                red_flags=["Analysis system error"],
                interview_questions=[],
                salary_estimate="Unable to estimate",
                fit_assessment="Manual review required",
                confidence_score=0.0,
                processing_time=time.time() - start_time
            )
    
    def batch_analyze_resumes(self, resumes: List[Dict], job_description: str = None) -> List[ResumeAnalysis]:
        """Analyze multiple resumes in batch"""
        
        print(f"\n📊 BATCH RESUME ANALYSIS")
        print("=" * 50)
        print(f"Processing {len(resumes)} resumes...")
        
        analyses = []
        
        for i, resume in enumerate(resumes, 1):
            print(f"\n🔍 Analyzing resume {i}/{len(resumes)}: {resume.get('name', f'Resume {i}')}")
            
            analysis = self.analyze_resume(
                resume['text'], 
                job_description, 
                resume.get('name', f'Candidate {i}')
            )
            
            analyses.append(analysis)
            
            print(f"   Overall Score: {analysis.overall_score:.1f}/100")
            print(f"   Fit Assessment: {analysis.fit_assessment}")
            print(f"   Processing Time: {analysis.processing_time:.2f}s")
        
        return analyses
    
    def compare_candidates(self, analyses: List[ResumeAnalysis]) -> Dict:
        """
        Compare multiple candidates and provide ranking
        """
        
        print(f"\n🏆 CANDIDATE COMPARISON ANALYSIS")
        print("=" * 50)
        
        # Sort by overall score
        sorted_analyses = sorted(analyses, key=lambda x: x.overall_score, reverse=True)
        
        comparison_data = {
            'ranking': [],
            'score_analysis': {},
            'skill_analysis': {},
            'recommendations': {}
        }
        
        # Create ranking
        for rank, analysis in enumerate(sorted_analyses, 1):
            comparison_data['ranking'].append({
                'rank': rank,
                'name': analysis.candidate_name,
                'overall_score': analysis.overall_score,
                'experience': analysis.years_of_experience,
                'fit_assessment': analysis.fit_assessment,
                'key_strengths': analysis.strengths[:3]  # Top 3 strengths
            })
        
        # Score analysis
        scores_df = pd.DataFrame([{
            'Name': a.candidate_name,
            'Overall': a.overall_score,
            'Experience': a.experience_score,
            'Education': a.education_score,
            'Skills': a.skills_score,
            'Communication': a.communication_score
        } for a in analyses])
        
        comparison_data['score_analysis'] = {
            'average_scores': scores_df.select_dtypes(include=[np.number]).mean().to_dict(),
            'score_ranges': {
                col: {'min': scores_df[col].min(), 'max': scores_df[col].max()}
                for col in scores_df.select_dtypes(include=[np.number]).columns
            }
        }
        
        # Skill analysis
        all_skills = []
        for analysis in analyses:
            all_skills.extend(analysis.key_skills)
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        comparison_data['skill_analysis'] = {
            'most_common_skills': dict(skill_counts.most_common(10)),
            'unique_skills_per_candidate': {
                a.candidate_name: len(set(a.key_skills)) for a in analyses
            }
        }
        
        # Recommendations
        top_candidates = sorted_analyses[:3]
        comparison_data['recommendations'] = {
            'top_3_candidates': [a.candidate_name for a in top_candidates],
            'interview_priority': top_candidates[0].candidate_name if top_candidates else None,
            'diversity_insights': self._analyze_diversity(analyses)
        }
        
        return comparison_data
    
    def _analyze_diversity(self, analyses: List[ResumeAnalysis]) -> Dict:
        """Analyze diversity aspects in candidate pool"""
        
        education_levels = [a.education_level for a in analyses]
        experience_ranges = []
        
        for analysis in analyses:
            exp = analysis.years_of_experience
            if exp <= 2:
                experience_ranges.append("Entry Level")
            elif exp <= 5:
                experience_ranges.append("Mid Level")
            elif exp <= 10:
                experience_ranges.append("Senior Level")
            else:
                experience_ranges.append("Executive Level")
        
        from collections import Counter
        
        return {
            'education_distribution': dict(Counter(education_levels)),
            'experience_distribution': dict(Counter(experience_ranges)),
            'skill_diversity_score': len(set([skill for a in analyses for skill in a.key_skills])) / len(analyses)
        }
    
    def visualize_analysis_results(self, analyses: List[ResumeAnalysis], 
                                 comparison_data: Dict = None):
        """
        Create comprehensive visualizations of resume analysis results
        """
        
        if not analyses:
            print("No analyses to visualize")
            return
        
        # Create subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overall Score Distribution
        ax1 = plt.subplot(3, 3, 1)
        scores = [a.overall_score for a in analyses]
        names = [a.candidate_name.split()[0] for a in analyses]  # First name only
        
        bars = ax1.bar(names, scores, color=['#2E8B57' if s >= 80 else '#FFD700' if s >= 60 else '#DC143C' for s in scores])
        ax1.set_title('Overall Candidate Scores', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Score (0-100)')
        ax1.set_ylim(0, 100)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Score Breakdown Radar Chart
        ax2 = plt.subplot(3, 3, 2, projection='polar')
        categories = ['Experience', 'Education', 'Skills', 'Communication']
        
        for i, analysis in enumerate(analyses):
            values = [analysis.experience_score, analysis.education_score, 
                     analysis.skills_score, analysis.communication_score]
            values += values[:1]  # Complete the circle
            
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, 
                    label=analysis.candidate_name.split()[0], alpha=0.7)
            ax2.fill(angles, values, alpha=0.1)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 100)
        ax2.set_title('Score Breakdown by Category', fontweight='bold', fontsize=14, pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 3. Experience vs Education Scatter
        ax3 = plt.subplot(3, 3, 3)
        x_vals = [a.years_of_experience for a in analyses]
        y_vals = [a.education_score for a in analyses]
        colors = [a.overall_score for a in analyses]
        
        scatter = ax3.scatter(x_vals, y_vals, c=colors, s=200, alpha=0.7, cmap='RdYlGn')
        ax3.set_xlabel('Years of Experience')
        ax3.set_ylabel('Education Score')
        ax3.set_title('Experience vs Education', fontweight='bold', fontsize=14)
        
        # Add candidate names as annotations
        for i, analysis in enumerate(analyses):
            ax3.annotate(analysis.candidate_name.split()[0], 
                        (x_vals[i], y_vals[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.colorbar(scatter, ax=ax3, label='Overall Score')
        
        # 4. Skills Analysis
        ax4 = plt.subplot(3, 3, 4)
        all_skills = []
        for analysis in analyses:
            all_skills.extend(analysis.key_skills)
        
        from collections import Counter
        skill_counts = Counter(all_skills)
        top_skills = dict(skill_counts.most_common(8))
        
        ax4.barh(list(top_skills.keys()), list(top_skills.values()), color='skyblue')
        ax4.set_title('Most Common Skills Across Candidates', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Frequency')
        
        # 5. Fit Assessment Distribution
        ax5 = plt.subplot(3, 3, 5)
        fit_assessments = [a.fit_assessment for a in analyses]
        fit_counts = Counter(fit_assessments)
        
        colors_fit = {'Excellent': '#2E8B57', 'Good': '#FFD700', 'Fair': '#FFA500', 'Poor': '#DC143C'}
        pie_colors = [colors_fit.get(fit, '#808080') for fit in fit_counts.keys()]
        
        ax5.pie(fit_counts.values(), labels=fit_counts.keys(), autopct='%1.0f%%',
               colors=pie_colors, startangle=90)
        ax5.set_title('Fit Assessment Distribution', fontweight='bold', fontsize=14)
        
        # 6. Processing Time Analysis
        ax6 = plt.subplot(3, 3, 6)
        processing_times = [a.processing_time for a in analyses]
        
        ax6.bar(names, processing_times, color='lightcoral')
        ax6.set_title('Processing Time by Candidate', fontweight='bold', fontsize=14)
        ax6.set_ylabel('Time (seconds)')
        
        # 7. Red Flags Analysis
        ax7 = plt.subplot(3, 3, 7)
        red_flag_counts = [len(a.red_flags) for a in analyses]
        
        colors_flags = ['red' if count > 0 else 'green' for count in red_flag_counts]
        ax7.bar(names, red_flag_counts, color=colors_flags)
        ax7.set_title('Red Flags Count per Candidate', fontweight='bold', fontsize=14)
        ax7.set_ylabel('Number of Red Flags')
        
        # 8. Confidence Score Distribution
        ax8 = plt.subplot(3, 3, 8)
        confidence_scores = [a.confidence_score for a in analyses]
        
        ax8.hist(confidence_scores, bins=10, alpha=0.7, color='purple', edgecolor='black')
        ax8.set_title('Model Confidence Distribution', fontweight='bold', fontsize=14)
        ax8.set_xlabel('Confidence Score')
        ax8.set_ylabel('Frequency')
        ax8.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidence_scores):.2f}')
        ax8.legend()
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate summary statistics
        avg_score = np.mean([a.overall_score for a in analyses])
        avg_experience = np.mean([a.years_of_experience for a in analyses])
        total_red_flags = sum(len(a.red_flags) for a in analyses)
        avg_processing_time = np.mean([a.processing_time for a in analyses])
        
        summary_text = f"""
ANALYSIS SUMMARY
{'='*30}

📊 Candidates Analyzed: {len(analyses)}
📈 Average Score: {avg_score:.1f}/100
👔 Average Experience: {avg_experience:.1f} years
⚠️  Total Red Flags: {total_red_flags}
⏱️  Avg Processing Time: {avg_processing_time:.2f}s

🏆 TOP PERFORMER:
{max(analyses, key=lambda x: x.overall_score).candidate_name}
Score: {max(analyses, key=lambda x: x.overall_score).overall_score:.1f}/100

🎯 RECOMMENDATION:
{comparison_data['recommendations']['interview_priority'] if comparison_data else 'See detailed analysis'}
"""
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/visualizations/comprehensive_resume_analysis.png",
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, analyses: List[ResumeAnalysis], 
                               comparison_data: Dict = None, 
                               job_description: str = None) -> str:
        """
        Generate comprehensive written report
        """
        
        report = f"""
AI RESUME REVIEW REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Candidates Analyzed: {len(analyses)}

EXECUTIVE SUMMARY
{'='*30}
"""
        
        if comparison_data:
            top_candidate = comparison_data['recommendations']['interview_priority']
            report += f"""
🎯 RECOMMENDED CANDIDATE: {top_candidate}
📊 AVERAGE SCORE: {np.mean([a.overall_score for a in analyses]):.1f}/100
🏆 SCORE RANGE: {min(a.overall_score for a in analyses):.1f} - {max(a.overall_score for a in analyses):.1f}
⚠️  TOTAL RED FLAGS: {sum(len(a.red_flags) for a in analyses)}

TOP 3 CANDIDATES FOR INTERVIEW:
"""
            for i, candidate in enumerate(comparison_data['recommendations']['top_3_candidates'], 1):
                analysis = next(a for a in analyses if a.candidate_name == candidate)
                report += f"  {i}. {candidate} (Score: {analysis.overall_score:.1f}/100)\n"
        
        report += f"""

DETAILED CANDIDATE ANALYSES
{'='*40}
"""
        
        # Sort candidates by score for reporting
        sorted_analyses = sorted(analyses, key=lambda x: x.overall_score, reverse=True)
        
        for i, analysis in enumerate(sorted_analyses, 1):
            report += f"""
{i}. {analysis.candidate_name.upper()}
{'-'*40}
Overall Score: {analysis.overall_score:.1f}/100
Fit Assessment: {analysis.fit_assessment}
Years of Experience: {analysis.years_of_experience}
Education: {analysis.education_level}
Salary Estimate: {analysis.salary_estimate}

SCORE BREAKDOWN:
  • Experience: {analysis.experience_score:.1f}/100
  • Education: {analysis.education_score:.1f}/100
  • Skills: {analysis.skills_score:.1f}/100
  • Communication: {analysis.communication_score:.1f}/100

STRENGTHS:
"""
            for strength in analysis.strengths:
                report += f"  ✓ {strength}\n"
            
            report += "\nAREAS FOR IMPROVEMENT:\n"
            for weakness in analysis.weaknesses:
                report += f"  ⚠ {weakness}\n"
            
            if analysis.red_flags:
                report += "\n🚨 RED FLAGS:\n"
                for flag in analysis.red_flags:
                    report += f"  ❌ {flag}\n"
            
            report += f"\nKEY SKILLS: {', '.join(analysis.key_skills[:5])}"
            report += f"\nPREVIOUS COMPANIES: {', '.join(analysis.previous_companies)}"
            
            report += f"\n\nRECOMMENDATIONS:\n"
            for rec in analysis.recommendations:
                report += f"  • {rec}\n"
            
            report += "\n" + "="*60 + "\n"
        
        # Add business insights
        if comparison_data:
            report += f"""
BUSINESS INSIGHTS & RECOMMENDATIONS
{'='*50}

SKILL ANALYSIS:
Most In-Demand Skills:
"""
            for skill, count in comparison_data['skill_analysis']['most_common_skills'].items():
                report += f"  • {skill}: {count} candidates\n"
            
            report += f"""
DIVERSITY ANALYSIS:
Education Distribution:
"""
            for edu, count in comparison_data['skill_analysis'].get('education_distribution', {}).items():
                report += f"  • {edu}: {count}\n"
            
            report += f"""
HIRING RECOMMENDATIONS:
1. IMMEDIATE ACTION: Interview {comparison_data['recommendations']['interview_priority']}
2. BACKUP CANDIDATES: {', '.join(comparison_data['recommendations']['top_3_candidates'][1:3])}
3. SKILL GAPS: Consider training programs for missing technical skills
4. PROCESS IMPROVEMENT: Average analysis time: {np.mean([a.processing_time for a in analyses]):.2f}s per resume

NEXT STEPS:
□ Schedule interviews with top 3 candidates
□ Prepare behavioral and technical interview questions
□ Check references for recommended candidates
□ Consider salary negotiation ranges
□ Plan onboarding for selected candidate
"""
        
        # Save report
        report_path = f"{self.output_dir}/detailed_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
        return report
    
    def run_complete_analysis(self, job_description: str = None):
        """
        Run complete resume analysis demonstration
        """
        
        print("🤖 AI RESUME REVIEWER: USING LLMS TO SOLVE REAL PROBLEMS")
        print("=" * 80)
        print("This demonstration shows:")
        print("1. Advanced prompt engineering for structured LLM outputs")
        print("2. Robust parsing and error handling for production systems")
        print("3. Comprehensive candidate analysis and comparison")
        print("4. Business intelligence from unstructured resume data")
        print("5. Scalable LLM application architecture")
        print("=" * 80)
        
        # Sample job description
        if not job_description:
            job_description = """
Senior Data Scientist Position

We are seeking a Senior Data Scientist to join our growing AI team. The ideal candidate will have:

REQUIRED QUALIFICATIONS:
• 5+ years of experience in data science or machine learning
• Master's degree in Computer Science, Statistics, or related field
• Expert-level Python programming skills
• Experience with ML frameworks (TensorFlow, PyTorch, Scikit-learn)
• Cloud platform experience (AWS, GCP, or Azure)
• Strong communication skills and ability to work with cross-functional teams

PREFERRED QUALIFICATIONS:
• PhD in relevant field
• MLOps and model deployment experience
• Experience with big data technologies (Spark, Hadoop)
• Leadership or mentoring experience
• Publications or contributions to open-source projects

RESPONSIBILITIES:
• Lead end-to-end ML projects from research to production
• Mentor junior data scientists and establish best practices
• Collaborate with engineering teams on scalable ML infrastructure
• Present insights and recommendations to executive leadership
• Stay current with latest developments in AI/ML field
"""
        
        print(f"\n📋 Job Description:")
        print(job_description[:300] + "..." if len(job_description) > 300 else job_description)
        
        # Create sample resumes
        sample_resumes = self.create_sample_resumes()
        print(f"\n📊 Created {len(sample_resumes)} sample resumes for analysis")
        
        # Analyze all resumes
        analyses = self.batch_analyze_resumes(sample_resumes, job_description)
        
        # Compare candidates
        comparison_data = self.compare_candidates(analyses)
        
        # Generate visualizations
        print(f"\n📈 Generating analysis visualizations...")
        self.visualize_analysis_results(analyses, comparison_data)
        
        # Generate detailed report
        print(f"\n📄 Generating comprehensive report...")
        report = self.generate_detailed_report(analyses, comparison_data, job_description)
        
        # Save structured data
        analysis_data = []
        for analysis in analyses:
            analysis_data.append(asdict(analysis))
        
        with open(f"{self.output_dir}/structured_analysis_data.json", 'w') as f:
            json.dump({
                'job_description': job_description,
                'analyses': analysis_data,
                'comparison_data': comparison_data,
                'summary_stats': {
                    'total_candidates': len(analyses),
                    'average_score': np.mean([a.overall_score for a in analyses]),
                    'processing_time_total': sum(a.processing_time for a in analyses),
                    'red_flags_total': sum(len(a.red_flags) for a in analyses)
                }
            }, f, indent=2)
        
        # Business impact summary
        print(f"\n💼 BUSINESS IMPACT ANALYSIS")
        print("=" * 50)
        
        total_processing_time = sum(a.processing_time for a in analyses)
        manual_time_estimate = len(analyses) * 30 * 60  # 30 minutes per resume manually
        time_saved = manual_time_estimate - total_processing_time
        cost_savings = (time_saved / 3600) * 50  # $50/hour recruiter cost
        
        print(f"⏱️  Processing Time: {total_processing_time:.1f} seconds")
        print(f"👤 Manual Time Estimate: {manual_time_estimate/60:.1f} minutes")  
        print(f"💰 Time Saved: {time_saved/60:.1f} minutes")
        print(f"💵 Cost Savings: ${cost_savings:.2f}")
        print(f"📈 Efficiency Gain: {(time_saved/manual_time_estimate)*100:.1f}%")
        
        print(f"\n🎯 RECOMMENDED NEXT STEPS:")
        if comparison_data['recommendations']['interview_priority']:
            print(f"1. Schedule interview with {comparison_data['recommendations']['interview_priority']}")
        print("2. Prepare customized interview questions based on analysis")
        print("3. Check references for top candidates")
        print("4. Plan technical assessment if needed")
        print("5. Consider salary negotiation strategy")
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📁 All results saved to: {self.output_dir}/")
        
        return analyses, comparison_data

if __name__ == "__main__":
    # Initialize resume reviewer
    reviewer = ResumeReviewer(llm_provider="openai")  # Change to "huggingface" if needed
    
    # Run complete analysis
    analyses, comparison_data = reviewer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("🎓 KEY LEARNING OUTCOMES:")
    print("1. Prompt engineering is crucial for reliable LLM applications")
    print("2. Structured output parsing enables scalable automation")
    print("3. Error handling and fallbacks ensure production readiness")
    print("4. LLMs can provide business insights from unstructured data")
    print("5. Proper evaluation metrics validate LLM application performance")
    print("\n🚀 NEXT STEPS:")
    print("• Integrate with ATS systems for real-world deployment")
    print("• Add bias detection and fairness metrics")
    print("• Implement active learning for continuous improvement")
    print("• Explore fine-tuning for domain-specific requirements")
    print("• Build web interface for HR team access")
    print("="*80)