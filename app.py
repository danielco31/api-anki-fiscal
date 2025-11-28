import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pinecone import Pinecone

app = Flask(__name__)
CORS(app) # Permite que o Anki acesse o servidor

# Pega as chaves das "Configurações Secretas" do servidor
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

# Configurações
genai.configure(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
# O nome tem que ser igual ao que você usou no indexador
index = pc.Index("anki-estudos") 

@app.route('/', methods=['GET'])
def home():
    return "Servidor Anki Fiscal Online! ⚖️ Disponível."

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        dados = request.json
        pergunta = dados.get('prompt')
        
        if not pergunta:
            return jsonify({"text": "Erro: Nenhuma pergunta recebida."}), 400

        # 1. Transforma a pergunta do Anki em números
        emb_pergunta = genai.embed_content(
            model="models/text-embedding-004",
            content=pergunta,
            task_type="retrieval_query"
        )['embedding']
        
        # 2. Busca no Pinecone os 5 trechos mais parecidos nos seus PDFs
        busca = index.query(
            vector=emb_pergunta,
            top_k=5,
            include_metadata=True
        )
        
        # 3. Monta o texto de apoio (Contexto)
        contexto = ""
        fontes = set()
        for match in busca['matches']:
            if 'text' in match['metadata']:
                contexto += match['metadata']['text'] + "\n---\n"
                # Aqui ele pega o nome original do arquivo (com acento) para te mostrar
                fontes.add(match['metadata']['source'])
        
        if not contexto:
            contexto = "Não encontrei informações exatas nos PDFs fornecidos."

        # 4. Manda pro Gemini responder
        # Usamos o Flash 2.0 ou 1.5 que é rápido e grátis
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt_final = f"""
        Você é um tutor especialista em concursos fiscais.
        Responda à pergunta abaixo usando OBRIGATORIAMENTE o contexto fornecido.
        
        CONTEXTO (Meus Livros/PDFs):
        {contexto}
        
        PERGUNTA:
        {pergunta}
        
        INSTRUÇÕES:
        - Responda de forma direta e didática.
        - Se a resposta estiver no contexto, explique e cite o conceito.
        - No final, liste as fontes: {list(fontes)}
        """
        
        resposta = model.generate_content(prompt_final)
        return jsonify({"text": resposta.text})

    except Exception as e:
        return jsonify({"text": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)