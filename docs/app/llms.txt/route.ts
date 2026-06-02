import { getLLMSIndexText } from "@/lib/llms-text";

export const revalidate = false;

export function GET() {
  return new Response(getLLMSIndexText(), {
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
    },
  });
}
