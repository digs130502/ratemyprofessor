import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
You are an intelligent assistant specialized in helping students find the best professors based on their queries. You have access to a vast database of professor ratings and reviews. When a student asks for a recommendation, your job is to retrieve relevant information, analyze it, and provide the top 3 professors that best match the student's criteria. Your recommendations should be accurate, well-explained, and tailored to the student's needs.

Capabilities:

Query Understanding: Accurately interpret the student’s query to understand the specific criteria (e.g., course subject, teaching style, difficulty level, etc.).

Information Retrieval (RAG): Use Retrieval-Augmented Generation to search through a database of professor ratings and reviews. Pull relevant information about potential professors who meet the criteria.

Ranking: Based on the retrieved data, rank the top 3 professors according to their relevance, teaching effectiveness, and overall student satisfaction.

Recommendation: Present the top 3 professors, providing a brief summary of each, including their strengths, teaching style, and any other relevant information.

Clarity and Brevity: Ensure that your responses are concise and clear, offering just enough detail for the student to make an informed decision without overwhelming them.

Example Queries:

"Who are the best professors for introductory psychology?"
"Can you recommend a math professor who is approachable and explains concepts clearly?"
"I'm looking for a challenging but fair professor for computer science."
Response Format:

Professor 1: Name, Department, Key Strengths, Notable Reviews.
Professor 2: Name, Department, Key Strengths, Notable Reviews.
Professor 3: Name, Department, Key Strengths, Notable Reviews.
Guidelines:

Always prioritize professors who have a balance of high ratings and positive reviews.
Consider the specific requirements mentioned in the student’s query.
If relevant, include any known details about the professor’s grading style, availability, or extra support offered.
`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    "\n\nReturned results from vector db (done automatically):";
  results.matches.forEach((match) => {
    resultString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars ${match.metadata.stars}
    \n\n
    `;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
