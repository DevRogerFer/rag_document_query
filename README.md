# 🤖 RAG Document Query — Consulta Inteligente de Documentos

![Status](https://img.shields.io/badge/status-ativo-22c55e?style=for-the-badge)
![Deploy](https://img.shields.io/badge/deploy-railway-7c3aed?style=for-the-badge&logo=railway&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-gpt--5.4--mini-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge)

## 📖 Sobre o Projeto

O RAG Document Query é uma aplicação web que combina **Azure OpenAI** e **Azure AI Search** para permitir consultas em linguagem natural sobre documentos PDF enviados pelo usuário.

O fluxo é simples: o usuário faz upload de um ou mais PDFs, a aplicação os fragmenta, gera embeddings vetoriais e os indexa no Azure AI Search. A cada nova pergunta, os trechos mais relevantes são recuperados e enviados ao modelo GPT para gerar uma resposta contextualizada.

- Upload de múltiplos PDFs via interface web
- Processamento e indexação automática com vetores de 3072 dimensões
- Consultas em linguagem natural com respostas baseadas exclusivamente no conteúdo enviado
- Isolamento total de sessão — o índice é limpo a cada novo upload

## ✨ Funcionalidades

### 📤 Upload e Processamento de Documentos

- Upload de múltiplos arquivos PDF (até 200MB cada)
- Extração de texto via `PyPDF2` com fallback para `PyPDFLoader`
- Fragmentação inteligente com `TokenTextSplitter`
- Barra de progresso e preview do conteúdo extraído

### 🔍 Indexação Vetorial

- Geração de embeddings com `text-embedding-3-large` (3072 dimensões)
- Indexação no Azure AI Search com busca por similaridade vetorial
- Limpeza automática do índice antes de cada novo processamento, garantindo isolamento total entre documentos

### 💬 Consulta com IA

- Interface de chat para perguntas em linguagem natural
- Pipeline RAG com LCEL (LangChain Expression Language)
- Recuperação dos trechos mais relevantes via busca vetorial
- Geração de resposta com `gpt-5.4-mini` baseada apenas no contexto recuperado

### ⚙️ Infraestrutura

- Suporte a dois recursos Azure OpenAI distintos (embeddings e chat em regiões separadas)
- Variáveis de ambiente via `.env` (sem credenciais no código)
- Deploy pronto para Railway com `Procfile`

## 🛠️ Tecnologias

| Tecnologia | Uso |
|---|---|
| [Streamlit](https://streamlit.io/) | Interface web |
| [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) | Embeddings (`text-embedding-3-large`) e Chat (`gpt-5.4-mini`) |
| [Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search) | Indexação e busca vetorial |
| [LangChain](https://www.langchain.com/) | Orquestração do pipeline RAG |
| [langchain-openai](https://pypi.org/project/langchain-openai/) | Integração LangChain com Azure OpenAI |
| [langchain-community](https://pypi.org/project/langchain-community/) | `AzureSearch`, `PyPDFLoader` |
| [azure-search-documents](https://pypi.org/project/azure-search-documents/) | `SearchIndexClient` para gerenciar índices |
| [PyPDF2](https://pypi.org/project/PyPDF2/) | Extração de texto de PDFs |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Carregamento de variáveis de ambiente |

## 🧱 Estrutura do Projeto

```
rag_document_query/
├── app.py              # Aplicação principal (UI + pipeline RAG)
├── requirements.txt    # Dependências Python
├── Procfile            # Configuração de deploy para Railway
├── .env                # Variáveis de ambiente (não versionado)
└── .gitignore
```

## 🔄 Arquitetura RAG

```
PDF Upload
    │
    ▼
PyPDF2 / PyPDFLoader
    │  (extração de texto)
    ▼
TokenTextSplitter
    │  (fragmentação em chunks)
    ▼
AzureOpenAIEmbeddings (text-embedding-3-large)
    │  (geração de vetores 3072d)
    ▼
Azure AI Search Index
    │
    │  ← Pergunta do usuário
    ▼
Busca por Similaridade Vetorial
    │  (k trechos mais relevantes)
    ▼
AzureChatOpenAI (gpt-5.4-mini)
    │  (geração de resposta contextualizada)
    ▼
Resposta ao Usuário
```

## ⚙️ Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Azure AI Search
AZURE_AI_SEARCH_SERVICE_NAME=https://<seu-servico>.search.windows.net
AZURE_AI_SEARCH_API_KEY=<sua-chave>
AZURE_AI_SEARCH_INDEX_NAME=<nome-do-indice>

# Azure OpenAI — Embeddings (ex: swedencentral)
AZURE_OPENAI_ENDPOINT=https://<seu-recurso-embeddings>.openai.azure.com/
AZURE_OPENAI_API_KEY=<chave-embeddings>
AZURE_EMBEDDING_MODEL=text-embedding-3-large

# Azure OpenAI — Chat (ex: northcentralus)
AZURE_OPENAI_CHAT_ENDPOINT=https://<seu-recurso-chat>.openai.azure.com/
AZURE_OPENAI_CHAT_API_KEY=<chave-chat>
AZURE_OPENAI_MODEL=gpt-5.4-mini
```

> **Nota:** Os recursos de embeddings e chat podem estar em regiões ou assinaturas Azure distintas. Configure os pares endpoint/chave separadamente.

## 🚀 Como Executar

### Pré-requisitos

- [Python](https://www.python.org/) 3.10+
- Recursos Azure: **Azure OpenAI** (embeddings + chat) e **Azure AI Search**

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/DevRogerFer/rag_document_query.git
cd rag_document_query

# 2. Criar e ativar o ambiente virtual
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar variáveis de ambiente
# Crie o arquivo .env conforme a seção acima

# 5. Iniciar a aplicação
streamlit run app.py
```

Acesse [http://localhost:8501](http://localhost:8501) no navegador.

## ☁️ Deploy

O projeto está configurado para deploy no Railway:

1. Conecte o repositório no [Railway](https://railway.app/)
2. Configure todas as variáveis de ambiente no painel do Railway
3. O deploy ocorre automaticamente a cada `git push` na branch `main`
4. O `Procfile` já instrui o Railway a iniciar o Streamlit na porta correta

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## 👨‍💻 Autor

Rogério Fernandes Siqueira

- LinkedIn: [linkedin.com/in/devrogerfer](https://www.linkedin.com/in/devrogerfer)
- GitHub: [github.com/DevRogerFer](https://github.com/DevRogerFer)

Se este projeto te ajudou ou inspirou, deixe uma estrela no repositório. ⭐
