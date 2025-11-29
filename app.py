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

        # ... (início do código igual) ...

        # ETAPA 3: AULA FINAL (COM OBEDIÊNCIA TOTAL AO GABARITO)
        prompt_final = f"""
        ATUE COMO: Tutor de Elite Multidisciplinar.
        CONTEXTO: Estudo Reverso.
        
        --- DADOS DO CARD (TEXTO + IMAGEM OCR) ---
        PERGUNTA DIGITADA: {pergunta_usuario}
        {dados_visuais}
        
        --- CONTEXTO DOS LIVROS ---
        {contexto}
        
        ⚠️ LEI MARCIAL DO GABARITO (LEIA COM ATENÇÃO):
        1. O card contém a RESPOSTA/GABARITO (seja no texto digitado ou no texto extraído da imagem).
        2. PROCURE por termos como: "Gabarito", "Resposta", "Comentários", "Letra X", "Correto/Errado".
        3. O gabarito fornecido no card é a VERDADE ABSOLUTA. Mesmo que você (IA) discorde, SIGA O GABARITO DO CARD.
        4. Se o card diz que a "II está errada", sua missão é explicar POR QUE ela está errada (busque a exceção ou a regra específica nos livros), e não tentar dizer que está certa.
        
        SUA MISSÃO:
        - Identifique o gabarito no texto acima.
        - Ministre uma MINI-AULA justificando ESSE gabarito.
        - No caso de Múltipla Escolha, explique o erro das alternativas que o gabarito diz estarem erradas.
        - OBRIGATÓRIO: Crie um EXEMPLO PRÁTICO.
        
        --- PERFIS ---
        [JURÍDICA/SUS] Cite a Lei/Norma.
        [SAÚDE] Explique mecanismo.
        [EXATAS] Mostre cálculo.
        
        --- AVISOS ---
        1. Corrija português.
        2. NÃO use LaTeX para texto.
        3. ⛔ NÃO liste fontes no final.
        """
        
        # ... (resto do código igual)
        
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
