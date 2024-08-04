import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrievalChain } from "langchain/chains/retrieval";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import * as dotenv from "dotenv";
dotenv.config(); // source env file

const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0.7, verbose: true });

const prompt = ChatPromptTemplate.fromTemplate(`
  Answer the user's question. 
  Context: {context}
  Question: {input}  
`);

const chain = await createStuffDocumentsChain({
  llm: model,
  prompt,
});

// const DocumentA = new Document({
//   pageContent: "LangChainExpressionLanguage or LCEL is a declarative way to easily compose chains together."
// })
// const response = await chain.invoke({
//   context: [ documentA ],
//   input: 'What is LCEL?',
// });

// LOAD DATA FROM WEBPAGE
const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/expression_language");
const docs = await loader.load();

// TRANSFROM DATA INTO PIECES
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200,
  chunkOverlap: 20,
});
const splitDocs = await splitter.splitDocuments(docs);

// EMBED DATA INTO STORE
const embeddings = new OpenAIEmbeddings();
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);

// RETRIEVE DATA
const retriever = vectorStore.asRetriever({
  k: 2,
});

const retrievalChain = await createRetrievalChain({
  combineDocsChain: chain,
  retriever,
});

// const response = await chain.invoke({
//   context: docs,
//   input: 'What is LCEL?',
// });

const response = await retrievalChain.invoke({
  input: 'What is LCEL?',
});

console.log(response);
console.log(response.answer);