import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";

dotenv.config();

const parser = new StringOutputParser();
const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0.7, maxTokens: 10, verbose: true });

// const prompt = ChatPromptTemplate.fromTemplate('You are a comedian. Tell a joke {input}')
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", "Translate the following into {language}:"],
  ["user", "{input}"],
]);

const chain = promptTemplate.pipe(model).pipe(parser);

const result = await chain.invoke({
  input: "dog",
});