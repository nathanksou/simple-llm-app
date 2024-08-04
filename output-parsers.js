import { StringOutputParser, CommaSeparatedListOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import { StructuredOutputParser } from "langchain/output_parsers";
import { z } from "zod";

dotenv.config();

const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0, maxTokens: 1000, verbose: true });

async function callStringOutputParser() {
  const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Translate the following into {language}:"],
    ["user", "{input}"],
  ]);
  const parser = new StringOutputParser();
  const chain = prompt.pipe(model).pipe(parser);
  
  return await chain.invoke({
    language: "italian",
    input: "happy",
  });
}

async function callListOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
    Provide 5 synonyms, separated by commas, for the following word {input}
    `);
  const parser = new CommaSeparatedListOutputParser();
  const chain = prompt.pipe(model).pipe(parser);
  
  return await chain.invoke({
    input: "happy",
  });
}

async function callStructuredOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
      Extract information from the following phrase.
      Formatting Instructions: {format_instructions}
      Phrase: {phrase}
    `);
  const parser = StructuredOutputParser.fromNamesAndDescriptions({
    name: "the name of the person",
    age: "the age of the person",
  });  
  const chain = prompt.pipe(model).pipe(parser);

  return await chain.invoke({
    format_instructions: parser.getFormatInstructions(),
    phrase: 'Max is 30 years old.',
  });
}

async function callZodOutputParser() {
  const prompt = ChatPromptTemplate.fromTemplate(`
      Extract information from the following phrase.
      Formatting Instructions: {format_instructions}
      Phrase: {phrase}
    `);
  const parser = StructuredOutputParser.fromZodSchema(
    z.object({
      recipe: z.string().describe("name of recipe"),
      ingredients: z.array(z.string()).describe("ingredients"),
    })
  );      
  const chain = prompt.pipe(model).pipe(parser);

  return await chain.invoke({
    format_instructions: parser.getFormatInstructions(),
    phrase: 'The ingredients for a Spaghetti Bolognese recipe are tomatoes, minced beef, garlic, wine, and herbs.',
  });
}

// const response = await callStringOutputParser();
// const response = await callListOutputParser();
// const response = await callStructuredOutputParser();
const response = await callZodOutputParser();
console.log(response);