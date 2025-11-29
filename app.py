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

model_vision = genai.GenerativeModel('gemini-2.0-flash') 

@app.route('/', methods=['GET'])
def home():
    return "Servidor RAG (Foco Aula Teórica) Online 🟢"

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        dados = request.json
        pergunta_usuario = dados.get('prompt', '')
        imagens = dados.get('images', []) 
        
        if not pergunta_usuario and not imagens:
            return jsonify({"text": "Erro: Card vazio."}), 400

        # ETAPA 1: VISÃO INTELIGENTE
        texto_para_busca = pergunta_usuario
        descricao_visual = "(Card puramente textual, sem imagens)."

        if imagens:
            try:
                img_bytes = base64.b64decode(imagens[0])
                prompt_ocr = """
                Analise esta imagem.
                1. Transcreva TODO o texto.
                2. Identifique o tipo (Certo/Errado ou Múltipla Escolha).
                3. Procure o GABARITO (marcações, texto verde).
                """
                resp_ocr = model_vision.generate_content([prompt_ocr, {'mime_type': 'image/jpeg', 'data': img_bytes}])
                texto_transcrito = resp_ocr.text
                descricao_visual = f"\n\n=== DADOS DA IMAGEM ===\n{texto_transcrito}"
                texto_para_busca += " " + texto_transcrito
            except Exception as e:
                print(f"Erro OCR: {e}")

        # ETAPA 2: BUSCA NO PINECONE
        contexto = "Sem referência nos PDFs."
        fontes = set()

        if texto_para_busca.strip():
            emb = genai.embed_content(model="models/text-embedding-004", content=texto_para_busca[:9000], task_type="retrieval_query")['embedding']
            busca = index.query(vector=emb, top_k=5, include_metadata=True)
            
            trechos = []
            for match in busca['matches']:
                if 'text' in match['metadata']:
                    trechos.append(match['metadata']['text'])
                    fonte = match['metadata'].get('source', 'Fonte Desconhecida')
                    fontes.add(fonte)
            if trechos: contexto = "\n---\n".join(trechos)

        # ETAPA 3: AULA FINAL (ESTRUTURA PEDAGÓGICA)
        prompt_final = f"""
        ATUE COMO: Tutor de Elite (Fiscal e Saúde).
        CONTEXTO: Estudo Reverso (Teoria a partir da Questão).
        
        DADOS DO CARD:
        {pergunta_usuario}
        {descricao_visual}
        
        CONTEXTO DOS LIVROS:
        {contexto}
        
        ⚠️ PRIORIDADE DE GABARITO:
        1. Ache a resposta correta nos dados do card.
        2. Assuma que ela é a Verdade Absoluta.
        
        SUA MISSÃO (SIGA ESTA ORDEM):
        
        1. 🎓 **MINI-AULA TEÓRICA:**
           - Antes de responder à questão, explique a TEORIA, o CONCEITO e a LEI por trás do assunto.
           - Ensine como se o aluno não soubesse nada sobre o tema.
        
        2. ✅ **RESOLUÇÃO DA QUESTÃO:**
           - Aplique a teoria explicada acima para justificar o gabarito.
           - Se for Múltipla Escolha, explique brevemente o erro das outras.
        
        3. 💡 **EXEMPLO PRÁTICO:**
           - Crie um caso concreto, clínico ou contábil para ilustrar.
        
        --- PERFIS ---
        [DIREITO/SUS] Cite a Lei.
        [SAÚDE] Explique o mecanismo.
        [EXATAS] Mostre o cálculo.
        
        --- AVISOS ---
        - Corrija português.
        - NÃO use LaTeX para texto.
        - NÃO liste fontes no final.
        """
        
        resposta = model_vision.generate_content(prompt_final)

        # ETAPA 4: RODAPÉ DE FONTES
        if not fontes:
            rodape_fontes = "\n\n<br><small><i>(Sem fontes exatas nos PDFs)</i></small>"
        else:
            lista_formatada = "<br>".join([f"• {f}" for f in fontes])
            rodape_fontes = f"\n\n<hr><b>📚 Fontes Consultadas:</b><br><small>{lista_formatada}</small>"
            
        texto_final = resposta.text + rodape_fontes
        return jsonify({"text": texto_final})

    except Exception as e:
        return jsonify({"text": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
