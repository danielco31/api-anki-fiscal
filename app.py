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
    return "Servidor RAG Adaptativo (Certo/Errado + Multipla) Online 🟢"

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        dados = request.json
        pergunta_usuario = dados.get('prompt', '')
        imagens = dados.get('images', []) 
        
        if not pergunta_usuario and not imagens:
            return jsonify({"text": "Erro: Card vazio."}), 400

        # ====================================================================
        # ETAPA 1: VISÃO INTELIGENTE (OCR FLEXÍVEL)
        # ====================================================================
        texto_para_busca = pergunta_usuario
        descricao_visual = ""

        if imagens:
            try:
                img_bytes = base64.b64decode(imagens[0])
                
                # Prompt que sabe lidar com QUALQUER formato
                prompt_ocr = """
                Analise esta imagem de estudo para concurso.
                
                SUA TAREFA DE EXTRAÇÃO:
                1. Transcreva TODO o texto visível (Enunciado + Itens).
                2. IDENTIFIQUE O TIPO: É Múltipla Escolha (A,B,C...)? É Certo/Errado (CEBRASPE)?
                3. PROCURE O GABARITO VISUAL: Procure por marcações, texto em verde, "Gabarito: X" ou comentários.
                
                Saída esperada:
                [TIPO DA QUESTÃO]: (Ex: Múltipla Escolha ou Certo/Errado)
                [TEXTO TRANSCRITO]: ...
                [GABARITO IDENTIFICADO NA IMAGEM]: (Se houver)
                """
                
                resp_ocr = model_vision.generate_content([
                    prompt_ocr,
                    {'mime_type': 'image/jpeg', 'data': img_bytes}
                ])
                
                texto_transcrito = resp_ocr.text
                descricao_visual = f"\n\n=== DADOS DA IMAGEM ===\n{texto_transcrito}"
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
        # ETAPA 3: AULA ADAPTATIVA (O Segredo está aqui)
        # ====================================================================
        prompt_final = f"""
        ATUE COMO: Tutor de Elite Multidisciplinar.
        CONTEXTO: Estudo Reverso.
        
        DADOS DO CARD (Frente + Verso + Imagem):
        {pergunta_usuario}
        {descricao_visual}
        
        CONTEXTO DOS LIVROS:
        {contexto}
        
        ⚠️ LÓGICA DE GABARITO (PRIORIDADE MÁXIMA):
        1. O usuário forneceu a resposta (no verso ou na imagem). ACHE ELA.
        2. Assuma que essa resposta está CERTA.
        3. Sua tarefa é JUSTIFICAR essa resposta com a teoria.
        
        SUA MISSÃO (Adapte-se ao formato encontrado):
        
        CASO A (CERTO / ERRADO):
        - Diga: "O item está [Certo/Errado] porque..."
        - Explique a pegadinha (se houver) ou confirme a teoria.
        
        CASO B (MÚLTIPLA ESCOLHA):
        - Diga: "A alternativa correta é a [Letra]..."
        - Explique o porquê da correta.
        - Brevemente, aponte o erro das outras (ex: "A letra A erra ao dizer...").
        
        CASO C (PERGUNTA ABERTA / CONCEITO):
        - Apenas explique o conceito de forma direta.
        
        OBRIGATÓRIO: Crie um EXEMPLO PRÁTICO no final.
        
        --- AVISOS ---
        1. Corrija português (palavras aglutinadas).
        2. NÃO use LaTeX para texto de lei.
        3. NÃO liste fontes no final.
        """
        
        resposta = model_vision.generate_content(prompt_final)

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
