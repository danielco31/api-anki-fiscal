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
        
        # 2. Busca no Pinecone os 5 trechos mais parecidos
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
                # Pega o nome da fonte. Se não tiver, usa "Desconhecido"
                fonte = match['metadata'].get('source', 'Fonte Desconhecida')
                fontes.add(fonte)
        
        if not contexto:
            contexto = "Não encontrei informações exatas nos PDFs fornecidos."

        # 4. Manda pro Gemini responder
        model = genai.GenerativeModel('gemini-2.0-flash') # 1.5 Flash é mais estável no Render
        
        prompt_final = f"""
        ATUE COMO: Um Tutor de Elite Multidisciplinar (Auditor Fiscal e Especialista em Saúde).
        CONTEXTO: O usuário faz "Estudo Reverso". O objetivo é dominar a teoria e saber aplicar na prática.
        
        SUA MISSÃO:
        1. Identifique a matéria.
        2. Ministre uma MINI-AULA teórica.
        3. OBRIGATÓRIO: Crie um EXEMPLO PRÁTICO/CONCRETO para ilustrar.
        
        --- PERFIS DE RESPOSTA (Adapte a didática) ---
        
        [PERFIL 1: JURÍDICA / SUS / HUMANAS]
        (Direito, Legislação do SUS, Auditoria, Português, Ética)
        - Teoria: Explique o conceito, a Lei (8.080, CF/88, LRF) ou a Norma.
        - 💡 EXEMPLO PRÁTICO: Crie uma situação hipotética (ex: "Imagine que o servidor João...", "Um paciente chega no posto de saúde e...").
        
        [PERFIL 2: SAÚDE / FARMÁCIA / BIOLÓGICAS]
        (Farmacologia, Química, Fisiologia, Patologia)
        - Teoria: Explique o mecanismo de ação, a interação ou a regra da Anvisa.
        - 💡 EXEMPLO PRÁTICO: Dê um exemplo clínico ou de rotina farmacêutica (ex: "Se um idoso tomar Digoxina com este fármaco, acontecerá X...", "Na indústria, essa reação é usada para...").
        
        [PERFIL 3: EXATAS / CONTABILIDADE / ECONOMIA]
        (Matemática, RLM, Estatística, Contabilidade, Economia)
        - Teoria: Explique a lógica e mostre o cálculo/lançamento passo a passo.
        - 💡 EXEMPLO PRÁTICO: Contextualize (ex: "A Empresa X comprou um caminhão...", "Para calcular os juros desse empréstimo...").
        
        [PERFIL 4: TECNOLOGIA (TI)]
        (Banco de Dados, SQL, Engenharia, Segurança)
        - Teoria: Explique a sintaxe ou arquitetura.
        - 💡 EXEMPLO PRÁTICO: Dê um caso de uso real (ex: "Um banco usa esse comando SQL para evitar que...").
        
        --- AVISOS DE QUALIDADE ---
        1. CORREÇÃO DE PORTUGUÊS: O contexto pode ter palavras aglutinadas ("palavrajunta"). Corrija o português ao explicar.
        2. FORMATAÇÃO (IMPORTANTE): NÃO use LaTeX/Matemática para escrever Leis ou Listas de Texto. Use apenas Markdown padrão.
        3. EXCEÇÃO: Use LaTeX apenas para Fórmulas Matemáticas reais e Cálculos.
        4. FONTE: Baseie-se no contexto recuperado abaixo.

        CONTEXTO RECUPERADO (Base de Conhecimento):
        {contexto}
        
        QUESTÃO/CARD DO ALUNO:
        {pergunta}
        """
        
        resposta = model.generate_content(prompt_final)

        # =================================================================
        # 5. A MÁGICA: O Python força a lista de fontes no final
        # =================================================================
        if not fontes:
            rodape_fontes = "\n\n<br><small><i>(Nenhuma fonte específica encontrada nos PDFs)</i></small>"
        else:
            # Cria a lista com bolinhas
            lista_formatada = "<br>".join([f"• {f}" for f in fontes])
            rodape_fontes = f"\n\n<hr><b>📚 Fontes Consultadas:</b><br><small>{lista_formatada}</small>"
            
        # Cola o rodapé no texto da IA
        texto_final = resposta.text + rodape_fontes
        
        return jsonify({"text": texto_final})

    except Exception as e:
        return jsonify({"text": f"Erro interno: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)