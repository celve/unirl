import { getLLMSFullText } from "@/lib/llms-text";

export const revalidate = false;

export async function GET() {
  return new Response(await getLLMSFullText(), {
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
    },
  });
}
