# Project Overview - Key Points

## 🎯 What is This Project?
An **end-to-end NLU Training Platform** combined with a **Rasa-powered Chatbot** system that manages the complete lifecycle: **Annotate → Train → Review → Retrain → Deploy → Chat**.

---

## 🔑 Key Points

### 1. **Dual-System Architecture**
   - **NLU Annotation Tool** (Flask-based backend + web frontend)
   - **Rasa Chatbot** (Express.js proxy + Rasa NLU engine)
   - Both systems work together for comprehensive chatbot development

### 2. **Workspace Management**
   - Multi-workspace support for managing multiple projects
   - Each workspace maintains its own:
     - Training data and annotations
     - Model versions (Rasa + spaCy)
     - Metadata and uncertain samples

### 3. **Complete Annotation Workflow**
   - Web-based annotation interface
   - Intent labeling and entity tagging
   - JSON format preview
   - Workspace-specific data storage

### 4. **Dual Model Training**
   - **spaCy NER**: Custom Named Entity Recognition
   - **Rasa NLU**: Intent classification and entity extraction
   - Version-controlled model storage

### 5. **Active Learning Module**
   - Detects low-confidence predictions (< 0.6 threshold)
   - Stores uncertain samples for re-annotation
   - Improves dataset quality over time

### 6. **Authentication & Security**
   - JWT-based authentication
   - User registration and login
   - Session validation
   - Protected endpoints

### 7. **Admin Dashboard**
   - Real-time statistics and metrics
   - Model health monitoring
   - Training history and versions
   - User management
   - Quick retrain actions

### 8. **Interactive Chatbot UI**
   - Custom e-commerce themed interface
   - Real-time conversation
   - Order tracking and product suggestions
   - Fallback handling

### 9. **Deployment Pipeline Visualization**
   - Visual CI/CD workflow
   - Docker, Container Registry, Cloud Deploy stages
   - Framework for future automation

---

## 🛠 Technology Stack

- **Backend**: Flask (NLU Tool) + Express.js (Rasa Proxy)
- **NLU/ML**: Rasa NLU, spaCy
- **Authentication**: JWT tokens
- **Frontend**: HTML, CSS, JavaScript
- **Chatbot**: Rasa Framework

---

## 📋 Main Use Cases

1. **Annotate training data** for chatbot intents and entities
2. **Train ML models** (spaCy and Rasa) from annotated data
3. **Review uncertain predictions** through active learning
4. **Monitor model performance** via admin dashboard
5. **Deploy and test** chatbot in interactive UI

---

## 🎓 Perfect For

- NLU/NLP learning and experimentation
- Chatbot development workflows
- Active learning implementation
- Multi-project management
- End-to-end ML pipeline demonstration
