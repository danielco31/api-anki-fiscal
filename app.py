import os
import base64 # <--- Importação Global
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pinecone import Pinecone

app = Flask(__name__)
CORS(app)

# CONFIGURAÇÕES
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("anki-estudos") 

# Define o modelo visual globalmente
model_vision = genai.GenerativeModel('gemini-2.0-flash') 

@app.route('/', methods=['GET'])
def home():
    return "Servidor RAG Online 🟢"

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        # Garante imports locais caso o global falhe
        import base64 
        
        dados = request.json
        pergunta_usuario = dados.get('prompt', '')
        imagens = dados.get('images', []) 
        
        if not pergunta_usuario and not imagens:
            return jsonify({"text": "Erro: Card vazio."}), 400

        # ====================================================================
        # ETAPA 1: VISÃO / OCR
        # ====================================================================
        texto_para_busca = pergunta_usuario
        dados_visuais = "(Este card é puramente textual, sem imagens)."

        if imagens:
            try:
                # Decodifica a imagem (Seguro)
                img_bytes = base64.b64decode(imagens[0])
                
                prompt_ocr = """
                Aja como um scanner.
                1. Transcreva TODO o texto da imagem.
                2. Se houver gabarito marcado, avise.
                3. Se for gráfico, descreva.
                """
                
                resp_ocr = model_vision.generate_content([
                    prompt_ocr,
                    {'mime_type': 'image/jpeg', 'data': img_bytes}
                ])
                
                texto_transcrito = resp_ocr.text
                dados_visuais = f"\n[CONTEÚDO DA IMAGEM]:\n{texto_transcrito}"
                texto_para_busca += " " + texto_transcrito
                
            except Exception as e:
                print(f"Erro OCR: {e}")

        # ====================================================================
        # ETAPA 2: BUSCA NO PINECONE
        # ====================================================================
        contexto = "Sem referência nos PDFs."
        fontes = set()

        if texto_para_busca.strip():
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
        # ETAPA 3: AULA FINAL
        # ====================================================================
        prompt_final = f"""
        ATUE COMO: Tutor de Elite Multidisciplinar.
        CONTEXTO: Estudo Reverso.
        
        --- DADOS DO CARD ---
        PERGUNTA: {pergunta_usuario}
        {dados_visuais}
        
        --- CONTEXTO DOS LIVROS ---
        {contexto}
        
        ⚠️ PRIORIDADE DE GABARITO:
        1. Procure a resposta/gabarito nos dados do card.
        2. Assuma que ela é a Verdade Absoluta.
        
        SUA MISSÃO:
        1. Ministre uma MINI-AULA teórica sobre o tema.
        2. Justifique o gabarito encontrado.
        3. OBRIGATÓRIO: Crie um EXEMPLO PRÁTICO.
        
        --- AVISOS ---
        - Corrija português.
        - NÃO use LaTeX para texto.
        - ⛔ NÃO liste fontes no final.
        """
        
        resposta = model_vision.generate_content(prompt_final)

        # ====================================================================
        # ETAPA 4: RODAPÉ DE FONTES
        # ====================================================================
        if not fontes:
            rodape_fontes = "\n\n<br><small><i>(Sem fontes exatas nos PDFs)</i></small>"
        else:
            lista_formatada = "<br>".join([f"• {f}" for f in fontes])
            rodape_fontes = f"\n\n<hr><b>📚 Fontes Consultadas:</b><br><small>{lista_formatada}</small>"
            
        texto_final = resposta.text + rodape_fontes
        
        return jsonify({"text": texto_final})

    except Exception as e:
        # Imprime o erro no console do Render para podermos ler
        print(f"ERRO CRÍTICO: {str(e)}") 
        return jsonify({"text": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
