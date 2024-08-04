import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";

dotenv.config(); // source env file

const parser = new StringOutputParser();
const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0.7, maxTokens: 10, verbose: true });

const messages = [
  new SystemMessage("Translate the following from English into Italian"),
  new HumanMessage("hi!"),
];

const chain = model.pipe(parser);
await chain.invoke(messages);
