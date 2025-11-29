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

model = genai.GenerativeModel('gemini-2.0-flash') 

@app.route('/', methods=['GET'])
def home():
    return "Servidor Híbrido (Texto & Visão) Online 🟢"

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        dados = request.json
        pergunta_usuario = dados.get('prompt', '')
        imagens = dados.get('images', []) 
        
        if not pergunta_usuario and not imagens:
            return jsonify({"text": "Erro: Card vazio."}), 400

        # ====================================================================
        # ETAPA 1: PROCESSAMENTO INTELIGENTE (COM OU SEM IMAGEM)
        # ====================================================================
        texto_para_busca = pergunta_usuario
        dados_visuais = "(Este card é puramente textual, sem imagens)."

        # SE TIVER IMAGEM: Faz OCR para enriquecer a busca
        if imagens:
            try:
                img_bytes = base64.b64decode(imagens[0])
                
                prompt_ocr = """
                ATENÇÃO: Extraia TODO o texto desta imagem.
                1. Se for questão, copie enunciado e alternativas.
                2. Se tiver gabarito marcado, indique.
                3. Se for gráfico/diagrama, descreva.
                """
                resp_ocr = model.generate_content([
                    prompt_ocr,
                    {'mime_type': 'image/jpeg', 'data': img_bytes}
                ])
                
                texto_transcrito = resp_ocr.text
                dados_visuais = f"\n[CONTEÚDO DA IMAGEM]:\n{texto_transcrito}"
                
                # A busca no Pinecone será: O que o usuário digitou + O que está na imagem
                texto_para_busca += " " + texto_transcrito
                
            except Exception as e:
                print(f"Erro no OCR (Ignorando imagem): {e}")

        # ====================================================================
        # ETAPA 2: BUSCA NO PINECONE (MEMÓRIA)
        # ====================================================================
        contexto = "Sem referência nos PDFs."
        fontes = set()

        if texto_para_busca.strip():
            # Corta texto muito longo para não travar o embedding
            emb = genai.embed_content(
                model="models/text-embedding-004",
                content=texto_para_busca[:9000], 
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
        # ETAPA 3: AULA FINAL (PROMPT ADAPTATIVO)
        # ====================================================================
        prompt_final = f"""
        ATUE COMO: Tutor de Elite Multidisciplinar (Auditor Fiscal e Especialista em Saúde).
        CONTEXTO: Estudo Reverso.
        
        --- DADOS DO CARD ---
        TEXTO DIGITADO: {pergunta_usuario}
        {dados_visuais}
        
        --- CONTEXTO DOS LIVROS (PINECONE) ---
        {contexto}
        
        ⚠️ LÓGICA DE GABARITO:
        1. Procure a resposta correta nos dados do card (Texto ou Imagem).
        2. Assuma que o gabarito fornecido está CERTO.
        3. Se não houver gabarito explícito, resolva a questão com base nos livros.
        
        SUA MISSÃO:
        - Ministre uma MINI-AULA teórica sobre o tema.
        - Se for questão, justifique o gabarito.
        - Se for conceito, explique profundamente.
        - OBRIGATÓRIO: Crie um EXEMPLO PRÁTICO.
        
        --- DIRETRIZES ---
        [DIREITO/SUS] Cite a Lei/Norma.
        [SAÚDE] Explique mecanismo/fisiopatologia.
        [EXATAS/TI] Mostre cálculo/lógica.
        
        AVISO: Corrija português e NÃO liste fontes no final.
        """
        
        resposta = model.generate_content(prompt_final)

        # ====================================================================
        # ETAPA 4: RODAPÉ DE FONTES (PYTHON)
        # ====================================================================
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
