#!/usr/bin/env python3
"""
AI Resume Reviewer - Capstone AI Product
Project 20 - PHASE 05: MLOps & Deployment

A complete AI-powered SaaS application for resume analysis and optimization.
Author: Your Name
Date: December 2024
"""

import os
import re
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# PDF processing
import PyPDF2
import pdfplumber

# NLP and ML
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeProcessor:
    """Process and extract text from resume PDFs"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            # Try with pdfplumber first (better formatting)
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    return self.clean_text(text)
            
            # Fallback to PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return self.clean_text(text)
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s@.-]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()

class SkillExtractor:
    """Extract skills from resume text using NLP"""
    
    def __init__(self):
        self.load_skill_database()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic extraction.")
            self.nlp = None
    
    def load_skill_database(self):
        """Load comprehensive skill database"""
        # Technical skills database
        self.technical_skills = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 
                'rust', 'kotlin', 'swift', 'typescript', 'sql', 'r', 'matlab', 'scala'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 
                'django', 'flask', 'spring', 'laravel', 'wordpress', 'bootstrap'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'oracle', 'sqlite', 'redis', 
                'cassandra', 'elasticsearch', 'dynamodb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform', 
                'jenkins', 'git', 'github', 'gitlab'
            ],
            'data_science': [
                'machine learning', 'deep learning', 'tensorflow', 'pytorch', 
                'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'tableau', 'power bi'
            ]
        }
        
        # Soft skills database
        self.soft_skills = [
            'leadership', 'communication', 'teamwork', 'problem solving', 
            'analytical thinking', 'creativity', 'adaptability', 'time management',
            'project management', 'collaboration', 'critical thinking', 'negotiation'
        ]
        
        # Create flat list of all skills
        self.all_skills = []
        for category in self.technical_skills.values():
            self.all_skills.extend(category)
        self.all_skills.extend(self.soft_skills)
        
        # Convert to lowercase for matching
        self.all_skills = [skill.lower() for skill in self.all_skills]
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = {
            'technical_skills': [],
            'soft_skills': [],
            'all_skills': []
        }
        
        # Extract technical skills by category
        for category, skills in self.technical_skills.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills['technical_skills'].append(skill)
                    found_skills['all_skills'].append(skill)
        
        # Extract soft skills
        for skill in self.soft_skills:
            if skill.lower() in text_lower:
                found_skills['soft_skills'].append(skill)
                found_skills['all_skills'].append(skill)
        
        # Remove duplicates
        for key in found_skills:
            found_skills[key] = list(set(found_skills[key]))
        
        return found_skills

class ExperienceAnalyzer:
    """Analyze work experience and achievements"""
    
    def __init__(self):
        self.experience_keywords = [
            'years', 'experience', 'worked', 'led', 'managed', 'developed', 
            'implemented', 'created', 'designed', 'improved', 'increased', 
            'decreased', 'optimized', 'achieved', 'delivered'
        ]
        
        self.achievement_patterns = [
            r'(\d+)%\s+(increase|improvement|reduction|growth)',
            r'saved\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'managed\s+(\d+)\s+people',
            r'led\s+team\s+of\s+(\d+)',
            r'(\d+)\s+years?\s+experience'
        ]
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience"""
        patterns = [
            r'(\d+)\+?\s+years?\s+of\s+experience',
            r'(\d+)\+?\s+years?\s+experience',
            r'experience:\s*(\d+)\+?\s+years?'
        ]
        
        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                try:
                    years = int(match)
                    max_years = max(max_years, years)
                except ValueError:
                    continue
        
        return max_years
    
    def extract_achievements(self, text: str) -> List[str]:
        """Extract quantifiable achievements"""
        achievements = []
        
        for pattern in self.achievement_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                achievements.append(str(match))
        
        return achievements
    
    def analyze_experience(self, text: str) -> Dict[str, Any]:
        """Comprehensive experience analysis"""
        return {
            'years_of_experience': self.extract_experience_years(text),
            'achievements': self.extract_achievements(text),
            'achievement_count': len(self.extract_achievements(text))
        }

class EducationAnalyzer:
    """Analyze educational background"""
    
    def __init__(self):
        self.degrees = [
            'phd', 'doctorate', 'ph.d', 'masters', 'master', 'mba', 'ms', 'ma', 
            'bachelor', 'ba', 'bs', 'bsc', 'associate', 'diploma', 'certificate'
        ]
        
        self.institutions = [
            'university', 'college', 'institute', 'school', 'academy'
        ]
    
    def extract_education(self, text: str) -> Dict[str, Any]:
        """Extract education information"""
        text_lower = text.lower()
        
        found_degrees = []
        for degree in self.degrees:
            if degree in text_lower:
                found_degrees.append(degree)
        
        # Calculate education score based on highest degree
        education_score = 0
        if any(d in found_degrees for d in ['phd', 'doctorate', 'ph.d']):
            education_score = 100
        elif any(d in found_degrees for d in ['masters', 'master', 'mba', 'ms', 'ma']):
            education_score = 85
        elif any(d in found_degrees for d in ['bachelor', 'ba', 'bs', 'bsc']):
            education_score = 70
        elif any(d in found_degrees for d in ['associate', 'diploma']):
            education_score = 50
        elif 'certificate' in found_degrees:
            education_score = 30
        
        return {
            'degrees': found_degrees,
            'education_score': education_score,
            'has_higher_education': education_score >= 50
        }

class FormatAnalyzer:
    """Analyze resume format and structure"""
    
    def __init__(self):
        self.ideal_sections = [
            'summary', 'objective', 'experience', 'education', 'skills', 
            'projects', 'certifications', 'awards', 'achievements'
        ]
    
    def analyze_format(self, text: str) -> Dict[str, Any]:
        """Analyze resume format and structure"""
        text_lower = text.lower()
        
        # Check for common sections
        found_sections = []
        for section in self.ideal_sections:
            if section in text_lower:
                found_sections.append(section)
        
        # Calculate format score
        section_score = (len(found_sections) / len(self.ideal_sections)) * 100
        
        # Check for contact information
        has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
        has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
        
        contact_score = 0
        if has_email:
            contact_score += 50
        if has_phone:
            contact_score += 50
        
        # Overall format score
        format_score = (section_score * 0.7 + contact_score * 0.3)
        
        return {
            'sections_found': found_sections,
            'section_count': len(found_sections),
            'has_email': has_email,
            'has_phone': has_phone,
            'format_score': format_score
        }

class ResumeScorer:
    """Calculate comprehensive resume scores"""
    
    def __init__(self):
        self.weights = {
            'skills_score': 0.30,
            'experience_score': 0.25,
            'education_score': 0.20,
            'format_score': 0.15,
            'achievements_score': 0.10
        }
    
    def calculate_skills_score(self, skills: Dict[str, List[str]]) -> float:
        """Calculate skills score"""
        technical_count = len(skills.get('technical_skills', []))
        soft_count = len(skills.get('soft_skills', []))
        
        # Score based on skill diversity and count
        technical_score = min(technical_count * 10, 70)  # Max 70 for technical
        soft_score = min(soft_count * 5, 30)  # Max 30 for soft skills
        
        return technical_score + soft_score
    
    def calculate_experience_score(self, experience: Dict[str, Any]) -> float:
        """Calculate experience score"""
        years = experience.get('years_of_experience', 0)
        achievements = experience.get('achievement_count', 0)
        
        years_score = min(years * 10, 70)  # Max 70 for years
        achievement_score = min(achievements * 10, 30)  # Max 30 for achievements
        
        return years_score + achievement_score
    
    def calculate_achievements_score(self, experience: Dict[str, Any]) -> float:
        """Calculate achievements score"""
        achievement_count = experience.get('achievement_count', 0)
        return min(achievement_count * 25, 100)
    
    def calculate_overall_score(self, analysis_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall resume score"""
        # Calculate individual scores
        skills_score = self.calculate_skills_score(analysis_results.get('skills', {}))
        experience_score = self.calculate_experience_score(analysis_results.get('experience', {}))
        education_score = analysis_results.get('education', {}).get('education_score', 0)
        format_score = analysis_results.get('format', {}).get('format_score', 0)
        achievements_score = self.calculate_achievements_score(analysis_results.get('experience', {}))
        
        # Calculate weighted overall score
        overall_score = (
            skills_score * self.weights['skills_score'] +
            experience_score * self.weights['experience_score'] +
            education_score * self.weights['education_score'] +
            format_score * self.weights['format_score'] +
            achievements_score * self.weights['achievements_score']
        )
        
        return {
            'overall_score': min(overall_score, 100),
            'skills_score': skills_score,
            'experience_score': experience_score,
            'education_score': education_score,
            'format_score': format_score,
            'achievements_score': achievements_score
        }

class RecommendationEngine:
    """Generate intelligent recommendations for resume improvement"""
    
    def generate_recommendations(self, analysis_results: Dict[str, Any], 
                               scores: Dict[str, float]) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Skills recommendations
        if scores['skills_score'] < 50:
            recommendations.append(
                "🔧 **Add more technical skills** - Include relevant programming languages, "
                "tools, and technologies for your target role."
            )
        
        # Experience recommendations
        if scores['experience_score'] < 60:
            if analysis_results.get('experience', {}).get('achievement_count', 0) < 2:
                recommendations.append(
                    "📈 **Quantify your achievements** - Add specific numbers, percentages, "
                    "and measurable results to demonstrate your impact."
                )
        
        # Education recommendations
        if scores['education_score'] < 40:
            recommendations.append(
                "🎓 **Highlight your education** - Include relevant coursework, "
                "certifications, or professional development."
            )
        
        # Format recommendations
        if scores['format_score'] < 70:
            format_analysis = analysis_results.get('format', {})
            if not format_analysis.get('has_email'):
                recommendations.append(
                    "📧 **Add contact information** - Include a professional email address."
                )
            if len(format_analysis.get('sections_found', [])) < 4:
                recommendations.append(
                    "📋 **Improve structure** - Add clear sections like Summary, Experience, "
                    "Skills, and Education."
                )
        
        # Overall recommendations
        if scores['overall_score'] < 60:
            recommendations.append(
                "⭐ **Professional summary** - Add a compelling summary that highlights "
                "your key strengths and career goals."
            )
        
        # Default recommendations if score is good
        if not recommendations:
            recommendations.append(
                "✅ **Great job!** Your resume looks solid. Consider tailoring it "
                "for specific job applications."
            )
        
        return recommendations

class ResumeAnalyzer:
    """Main resume analysis orchestrator"""
    
    def __init__(self):
        self.processor = ResumeProcessor()
        self.skill_extractor = SkillExtractor()
        self.experience_analyzer = ExperienceAnalyzer()
        self.education_analyzer = EducationAnalyzer()
        self.format_analyzer = FormatAnalyzer()
        self.scorer = ResumeScorer()
        self.recommender = RecommendationEngine()
    
    def analyze_resume(self, uploaded_file) -> Dict[str, Any]:
        """Complete resume analysis pipeline"""
        try:
            # Extract text
            resume_text = self.processor.extract_text_from_pdf(uploaded_file)
            
            if not resume_text:
                return {'error': 'Could not extract text from the PDF'}
            
            # Perform analyses
            skills_analysis = self.skill_extractor.extract_skills(resume_text)
            experience_analysis = self.experience_analyzer.analyze_experience(resume_text)
            education_analysis = self.education_analyzer.extract_education(resume_text)
            format_analysis = self.format_analyzer.analyze_format(resume_text)
            
            # Compile analysis results
            analysis_results = {
                'text_length': len(resume_text),
                'word_count': len(resume_text.split()),
                'skills': skills_analysis,
                'experience': experience_analysis,
                'education': education_analysis,
                'format': format_analysis,
                'raw_text': resume_text[:1000] + "..." if len(resume_text) > 1000 else resume_text
            }
            
            # Calculate scores
            scores = self.scorer.calculate_overall_score(analysis_results)
            analysis_results['scores'] = scores
            
            # Generate recommendations
            recommendations = self.recommender.generate_recommendations(analysis_results, scores)
            analysis_results['recommendations'] = recommendations
            
            # Add timestamp
            analysis_results['analysis_timestamp'] = datetime.now().isoformat()
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Resume analysis failed: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}

def create_score_gauge(score: float, title: str) -> go.Figure:
    """Create a gauge chart for scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_skills_chart(skills_data: Dict[str, List[str]]) -> go.Figure:
    """Create skills visualization"""
    technical_count = len(skills_data.get('technical_skills', []))
    soft_count = len(skills_data.get('soft_skills', []))
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Technical Skills', 'Soft Skills'],
            y=[technical_count, soft_count],
            marker_color=['blue', 'green']
        )
    ])
    
    fig.update_layout(
        title='Skills Distribution',
        xaxis_title='Skill Type',
        yaxis_title='Number of Skills',
        height=400
    )
    
    return fig

def create_scores_radar(scores: Dict[str, float]) -> go.Figure:
    """Create radar chart for all scores"""
    categories = ['Skills', 'Experience', 'Education', 'Format', 'Achievements']
    values = [
        scores.get('skills_score', 0),
        scores.get('experience_score', 0),
        scores.get('education_score', 0),
        scores.get('format_score', 0),
        scores.get('achievements_score', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Resume'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title="Resume Score Breakdown",
        height=500
    )
    
    return fig

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Resume Reviewer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🤖 AI Resume Reviewer")
    st.subheader("Get instant feedback and improve your resume with AI")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ResumeAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.write("1. Upload your resume (PDF format)")
        st.write("2. Get instant AI analysis")
        st.write("3. Review recommendations")
        st.write("4. Download detailed report")
        
        st.header("🎯 What We Analyze")
        st.write("• Skills extraction")
        st.write("• Experience evaluation")
        st.write("• Education assessment")
        st.write("• Format analysis")
        st.write("• Achievement quantification")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose your resume (PDF)",
        type=['pdf'],
        help="Upload a PDF file of your resume"
    )
    
    if uploaded_file is not None:
        with st.spinner('🔍 Analyzing your resume...'):
            # Analyze resume
            results = st.session_state.analyzer.analyze_resume(uploaded_file)
            
            if 'error' in results:
                st.error(f"Error: {results['error']}")
                return
            
            # Display results
            st.success("✅ Analysis complete!")
            
            # Overall score
            overall_score = results['scores']['overall_score']
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                fig_gauge = create_score_gauge(overall_score, "Overall Resume Score")
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.metric("Word Count", results['word_count'])
                st.metric("Technical Skills", len(results['skills']['technical_skills']))
            
            with col3:
                st.metric("Years Experience", results['experience']['years_of_experience'])
                st.metric("Achievements", results['experience']['achievement_count'])
            
            # Detailed scores
            st.header("📊 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Radar chart
                fig_radar = create_scores_radar(results['scores'])
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col2:
                # Skills chart
                fig_skills = create_skills_chart(results['skills'])
                st.plotly_chart(fig_skills, use_container_width=True)
            
            # Recommendations
            st.header("💡 Recommendations")
            for i, recommendation in enumerate(results['recommendations'], 1):
                st.write(f"{i}. {recommendation}")
            
            # Detailed breakdown
            with st.expander("📋 Detailed Breakdown"):
                
                # Skills
                st.subheader("🔧 Skills Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Technical Skills:**")
                    if results['skills']['technical_skills']:
                        for skill in results['skills']['technical_skills']:
                            st.write(f"• {skill}")
                    else:
                        st.write("None detected")
                
                with col2:
                    st.write("**Soft Skills:**")
                    if results['skills']['soft_skills']:
                        for skill in results['skills']['soft_skills']:
                            st.write(f"• {skill}")
                    else:
                        st.write("None detected")
                
                # Experience
                st.subheader("💼 Experience Analysis")
                st.write(f"• Years of Experience: {results['experience']['years_of_experience']}")
                st.write(f"• Quantified Achievements: {results['experience']['achievement_count']}")
                
                # Education
                st.subheader("🎓 Education Analysis")
                education = results['education']
                st.write(f"• Education Score: {education['education_score']}/100")
                st.write(f"• Degrees Found: {', '.join(education['degrees']) if education['degrees'] else 'None'}")
                
                # Format
                st.subheader("📄 Format Analysis")
                format_info = results['format']
                st.write(f"• Format Score: {format_info['format_score']:.1f}/100")
                st.write(f"• Sections Found: {', '.join(format_info['sections_found'])}")
                st.write(f"• Has Email: {'✅' if format_info['has_email'] else '❌'}")
                st.write(f"• Has Phone: {'✅' if format_info['has_phone'] else '❌'}")
            
            # Export results
            st.header("📥 Export Results")
            
            # Create summary report
            report_data = {
                'filename': uploaded_file.name,
                'analysis_date': results['analysis_timestamp'],
                'overall_score': overall_score,
                'scores': results['scores'],
                'recommendations': results['recommendations'],
                'skills_count': len(results['skills']['all_skills']),
                'experience_years': results['experience']['years_of_experience']
            }
            
            report_json = json.dumps(report_data, indent=2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📄 Download JSON Report",
                    data=report_json,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                # Create PDF report (placeholder)
                st.info("📋 PDF report feature coming soon!")
    
    else:
        # Welcome message
        st.info("👆 Please upload your resume PDF to get started!")
        
        # Demo information
        st.header("🚀 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🔍 AI Analysis")
            st.write("• Skill extraction")
            st.write("• Experience analysis")
            st.write("• Education assessment")
        
        with col2:
            st.subheader("📊 Smart Scoring")
            st.write("• Overall compatibility")
            st.write("• Section-wise scores")
            st.write("• ATS optimization")
        
        with col3:
            st.subheader("💡 Recommendations")
            st.write("• Personalized feedback")
            st.write("• Improvement suggestions")
            st.write("• Industry best practices")

if __name__ == "__main__":
    main()