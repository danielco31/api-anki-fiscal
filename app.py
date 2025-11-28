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

# Modelo para OCR e Resposta
model_vision = genai.GenerativeModel('gemini-2.0-flash') 

@app.route('/', methods=['GET'])
def home():
    return "Servidor RAG (Gabarito Fixo) Online 🟢"

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        dados = request.json
        pergunta_usuario = dados.get('prompt', '')
        imagens = dados.get('images', []) 
        
        if not pergunta_usuario and not imagens:
            return jsonify({"text": "Erro: Card vazio."}), 400

        # ====================================================================
        # ETAPA 1: VISÃO (OCR) - Extrai texto da imagem para achar o Gabarito
        # ====================================================================
        texto_para_busca = pergunta_usuario
        descricao_visual = ""

        if imagens:
            img_bytes = base64.b64decode(imagens[0])
            
            # Pede para a IA ler TUDO (incluindo o gabarito se estiver na imagem)
            prompt_ocr = "Transcreva TODO o texto desta imagem. Se houver gabarito ou resposta marcada, transcreva também."
            
            resp_ocr = model_vision.generate_content([
                prompt_ocr,
                {'mime_type': 'image/jpeg', 'data': img_bytes}
            ])
            
            texto_transcrito = resp_ocr.text
            descricao_visual = f"\n\n[CONTEÚDO DA IMAGEM]:\n{texto_transcrito}"
            texto_para_busca += " " + texto_transcrito

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
        # ETAPA 3: AULA COM GABARITO FIXO (GEMINI)
        # ====================================================================
        prompt_final = f"""
        ATUE COMO: Um Tutor de Elite Multidisciplinar (Auditor Fiscal e Especialista em Saúde).
        CONTEXTO: O usuário faz "Estudo Reverso". Ele envia a PERGUNTA e a RESPOSTA juntas.
        
        DADOS DO CARD (Texto + Imagem Transcrita):
        {pergunta_usuario}
        {descricao_visual}
        
        CONTEXTO RECUPERADO DOS LIVROS:
        {contexto}
        
        ⚠️ REGRA DE OURO DO GABARITO (CRÍTICO):
        1. NÃO tente resolver a questão sozinho.
        2. O Gabarito Correto JÁ ESTÁ nos dados do card acima (Procure por "Gabarito", "Letra", "Certo/Errado" ou texto explicativo).
        3. Assuma que o gabarito fornecido pelo aluno é a Verdade Absoluta.
        4. Sua tarefa é EXPLICAR POR QUE aquele gabarito está certo, usando a teoria dos livros.
        
        SUA MISSÃO:
        1. Identifique a matéria e o Perfil.
        2. Ministre uma MINI-AULA teórica justificando a resposta do aluno.
        3. OBRIGATÓRIO: Crie um EXEMPLO PRÁTICO.
        
        --- PERFIS DE RESPOSTA ---
        
        [PERFIL 1: JURÍDICA / SUS / HUMANAS]
        - Teoria: Explique o conceito e cite a Lei/Norma (8.080, CF/88, LRF).
        - Exemplo: Crie uma situação hipotética ("Imagine que o servidor João...").
        
        [PERFIL 2: SAÚDE / FARMÁCIA]
        - Teoria: Explique mecanismo de ação, interação ou regra da Anvisa.
        - Exemplo: Dê um exemplo clínico.
        
        [PERFIL 3: EXATAS / CONTABILIDADE]
        - Ação: Mostre o CÁLCULO ou LANÇAMENTO que chega no resultado do gabarito.
        
        [PERFIL 4: TI]
        - Ação: Explique a lógica do código ou diagrama.
        
        --- AVISOS ---
        1. Corrija português (palavras aglutinadas).
        2. NÃO use LaTeX para texto.
        3. NÃO liste as fontes no final (o sistema já faz isso).
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