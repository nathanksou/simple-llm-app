import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';

import * as dotenv from 'dotenv';

dotenv.config(); // source env file

// Load data and create vector store
const createVectorStore = async () => {
  const loader = new CheerioWebBaseLoader('https://js.langchain.com/docs/expression_language');
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  return vectorStore;
};

// Create retrieval chain
const createChain = async (vectorStore) => {
  const model = new ChatOpenAI({ modelName: 'gpt-3.5-turbo', temperature: 0.7, verbose: true });

  const prompt = ChatPromptTemplate.fromMessages([
    ['system', "Answer the user's questions based on the following context: {context}."],
    new MessagesPlaceholder('chat_history'),
    ['user', '{input}'],
  ]);
  
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });
  
  const retriever = vectorStore.asRetriever({
    k: 2,
  });

  const retrieverPrompt = ChatPromptTemplate.fromMessages([
    new MessagesPlaceholder('chat_history'),
    ['user', '{input}'],
    ['user', 'Given the above conversation, generate a search query to look up in order to get information relevant to the conversation'],
  ]);

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: retrieverPrompt,
  });

  const conversationChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever: historyAwareRetriever,
  });

  return conversationChain;
}

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

const response = await chain.invoke({
  input: 'What is LCEL?',
  chat_history: [
    new HumanMessage('Hello'),
    new AIMessage('Hi, how can I help you?'),
    new HumanMessage('My name is Leon'),
    new AIMessage('Pleasure Leon, what can I do for you?'),
  ],
});

console.log(response);
console.log(response.answer);