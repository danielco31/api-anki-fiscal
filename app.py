import os
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

@app.route('/', methods=['GET'])
def home():
    return "Servidor Universal (Vision + RAG) Online 🟢"

@app.route('/perguntar', methods=['POST'])
def perguntar():
    try:
        dados = request.json
        pergunta = dados.get('prompt', '')
        # AQUI A MUDANÇA: Captura as imagens enviadas pelo Anki
        imagens = dados.get('images', []) 
        
        if not pergunta and not imagens:
            return jsonify({"text": "Erro: Card vazio."}), 400

        # 1. Busca Semântica no Pinecone (Usa o texto para achar o livro)
        contexto = "Sem referência nos PDFs (Card visual ou sem texto)."
        fontes = set()

        if pergunta.strip():
            emb_pergunta = genai.embed_content(
                model="models/text-embedding-004",
                content=pergunta,
                task_type="retrieval_query"
            )['embedding']
            
            busca = index.query(vector=emb_pergunta, top_k=5, include_metadata=True)
            
            trechos = []
            for match in busca['matches']:
                if 'text' in match['metadata']:
                    trechos.append(match['metadata']['text'])
                    fonte = match['metadata'].get('source', 'Fonte Desconhecida')
                    fontes.add(fonte)
            
            if trechos:
                contexto = "\n---\n".join(trechos)

        # 2. Configura o Modelo (1.5 Flash é ótimo para visão e texto)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt_final = f"""
        ATUE COMO: Um Tutor de Elite Multidisciplinar (Auditor Fiscal e Especialista em Saúde).
        CONTEXTO: O usuário faz "Estudo Reverso" com apoio visual.
        
        SUA MISSÃO:
        1. Se houver IMAGEM (Gráfico, Tabela, Diagrama, Sintaxe): Analise-a detalhadamente.
        2. Identifique a matéria e o Perfil (abaixo).
        3. Ministre uma MINI-AULA teórica conectando a Imagem (se houver) ao Contexto dos livros.
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
        - Visão: Se houver gráfico, explique os eixos e o deslocamento das curvas.
        - Ação: Mostre o CÁLCULO passo a passo ou o LANÇAMENTO (D/C).
        
        [PERFIL 4: TECNOLOGIA (TI)]
        (Banco de Dados, SQL, Engenharia)
        - Visão: Se houver diagrama ER ou código, explique a lógica e o fluxo.
        
        --- AVISOS DE QUALIDADE ---
        1. CORREÇÃO DE PORTUGUÊS: Corrija palavras aglutinadas do contexto.
        2. FORMATAÇÃO: NÃO use LaTeX para texto comum. Use apenas para cálculos.
        3. FONTE: Baseie-se no contexto recuperado.

        CONTEXTO RECUPERADO (Base de Conhecimento):
        {contexto}
        
        QUESTÃO/CARD DO ALUNO:
        {pergunta}

        ⚠️ REGRA DE OURO (FORMATAÇÃO):
        - NÃO escreva "Fontes:" ou liste os arquivos no final da sua resposta. 
        - O sistema já fará essa listagem automaticamente.
        """
        
        # 3. Monta o "Pacote Misto" (Texto + Imagens) para o Gemini
        conteudo_envio = [prompt_final]
        
        for img_b64 in imagens:
            # Adiciona cada imagem como um objeto Blob
            conteudo_envio.append({'mime_type': 'image/jpeg', 'data': img_b64})
            
        resposta = model.generate_content(conteudo_envio)

        # 4. A MÁGICA: O Python força a lista de fontes no final
        if not fontes:
            rodape_fontes = "\n\n<br><small><i>(Sem fontes nos PDFs)</i></small>"
        else:
            lista_formatada = "<br>".join([f"• {f}" for f in fontes])
            rodape_fontes = f"\n\n<hr><b>📚 Fontes Consultadas:</b><br><small>{lista_formatada}</small>"
            
        texto_final = resposta.text + rodape_fontes
        
        return jsonify({"text": texto_final})

    except Exception as e:
        return jsonify({"text": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))