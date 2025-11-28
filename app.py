import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pinecone import Pinecone

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("anki-estudos") 

# Modelo rápido para transcrição e embedding
model_vision = genai.GenerativeModel('gemini-2.0-flash') 

@app.route('/', methods=['GET'])
def home():
    return "Servidor RAG Vision OCR Online 👁️📚"

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        dados = request.json
        pergunta_usuario = dados.get('prompt', '')
        imagens = dados.get('images', []) 
        
        if not pergunta_usuario and not imagens:
            return jsonify({"text": "Erro: Card vazio."}), 400

        # ====================================================================
        # ETAPA 1: PRÉ-LEITURA (OCR)
        # Se tiver imagem, extrai o texto dela para poder buscar no Pinecone
        # ====================================================================
        texto_para_busca = pergunta_usuario
        descricao_visual = ""

        if imagens:
            # Pega a primeira imagem para análise (geralmente é o print da questão)
            img_bytes = base64.b64decode(imagens[0])
            
            # Pede ao Gemini para transcrever o que vê
            prompt_ocr = "Transcreva TODO o texto presente nesta imagem. Se houver gráfico ou diagrama, descreva o que ele representa em detalhes."
            
            resp_ocr = model_vision.generate_content([
                prompt_ocr,
                {'mime_type': 'image/jpeg', 'data': img_bytes}
            ])
            
            texto_transcrito = resp_ocr.text
            descricao_visual = f"\n\n[CONTEÚDO VISUAL TRANSCRITO DA IMAGEM]:\n{texto_transcrito}"
            
            # Enriquece o texto de busca: Pergunta digitada + Texto da imagem
            texto_para_busca += " " + texto_transcrito

        # ====================================================================
        # ETAPA 2: BUSCA NA BIBLIOTECA (PINECONE)
        # Agora o Pinecone recebe o texto da imagem e consegue achar o livro!
        # ====================================================================
        contexto = "Sem referência nos PDFs."
        fontes = set()

        if texto_para_busca.strip():
            emb = genai.embed_content(
                model="models/text-embedding-004",
                content=texto_para_busca[:9000], # Limite de segurança para embedding
                task_type="retrieval_query"
            )['embedding']
            
            busca = index.query(vector=emb, top_k=5, include_metadata=True)
            
            trechos = []
            for match in busca['matches']:
                if 'text' in match['metadata']:
                    trechos.append(match['metadata']['text'])
                    fonte = match['metadata'].get('source', 'Fonte Desconhecida')
                    fontes.add(fonte)
            
            if trechos:
                contexto = "\n---\n".join(trechos)

        # ====================================================================
        # ETAPA 3: AULA FINAL (GEMINI)
        # ====================================================================
        prompt_final = f"""
        ATUE COMO: Um Tutor de Elite Multidisciplinar (Auditor Fiscal e Especialista em Saúde).
        CONTEXTO: O usuário faz "Estudo Reverso". O objetivo é dominar a teoria e saber aplicar na prática.
        
        INFORMAÇÃO DO CARD:
        {pergunta_usuario}
        {descricao_visual}
        
        CONTEXTO RECUPERADO DOS LIVROS:
        {contexto}
        
        SUA MISSÃO:
        1. Identifique a matéria e o Perfil (abaixo).
        2. Analise o conteúdo (texto + imagem transcrita) detalhadamente.
        3. Ministre uma MINI-AULA teórica conectando a questão ao Contexto dos livros.
        4. OBRIGATÓRIO: Crie um EXEMPLO PRÁTICO.
        
        --- PERFIS DE RESPOSTA (Adapte a didática) ---
        
        [PERFIL 1: JURÍDICA / SUS / HUMANAS]
        (Direito, Legislação do SUS, Auditoria, Português)
        - Teoria: Explique o conceito, a Lei (8.080, CF/88, LRF) ou a Norma.
        - Exemplo: Crie uma situação hipotética (ex: "Imagine que o servidor João...").
        
        [PERFIL 2: SAÚDE / FARMÁCIA / BIOLÓGICAS]
        (Farmacologia, Química, Fisiologia)
        - Teoria: Explique o mecanismo de ação, interação ou regra da Anvisa.
        - Exemplo: Dê um exemplo clínico (ex: "Se um paciente idoso tomar...").
        
        [PERFIL 3: EXATAS / CONTABILIDADE / ECONOMIA]
        (Matemática, RLM, Contabilidade, Economia)
        - Análise: Explique os eixos do gráfico ou a lógica matemática.
        - Ação: Mostre o CÁLCULO passo a passo ou o LANÇAMENTO (D/C).
        
        [PERFIL 4: TECNOLOGIA (TI)]
        (Banco de Dados, SQL, Engenharia)
        - Análise: Explique o diagrama ou código.
        
        --- AVISOS DE QUALIDADE ---
        1. CORREÇÃO DE PORTUGUÊS: Corrija palavras aglutinadas do contexto.
        2. FORMATAÇÃO: NÃO use LaTeX para texto comum. Use apenas para cálculos.
        3. FONTE: Baseie-se no contexto recuperado.

        ⚠️ REGRA DE OURO (FORMATAÇÃO):
        - NÃO escreva "Fontes:" ou liste os arquivos no final da sua resposta. 
        - O sistema já fará essa listagem automaticamente.
        """
        
        # Envia apenas texto (já que a imagem foi transcrita na etapa 1)
        # Isso economiza tokens e mantém o foco no contexto recuperado
        resposta = model_vision.generate_content(prompt_final)

        # ====================================================================
        # ETAPA 4: RODAPÉ DE FONTES (PYTHON)
        # ====================================================================
        if not fontes:
            rodape_fontes = "\n\n<br><small><i>(Sem fontes exatas nos PDFs para esta imagem)</i></small>"
        else:
            lista_formatada = "<br>".join([f"• {f}" for f in fontes])
            rodape_fontes = f"\n\n<hr><b>📚 Fontes Consultadas:</b><br><small>{lista_formatada}</small>"
            
        texto_final = resposta.text + rodape_fontes
        
        return jsonify({"text": texto_final})

    except Exception as e:
        return jsonify({"text": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))