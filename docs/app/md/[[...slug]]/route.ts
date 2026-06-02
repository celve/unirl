import { getLLMText } from "@/lib/get-llm-text";
import { source } from "@/lib/source";

type MarkdownRouteContext = {
  params: Promise<{
    slug?: string[];
  }>;
};

export const dynamicParams = false;
export const revalidate = false;

export function generateStaticParams() {
  return source.getPages("en").map((page) => ({
    slug: [...page.slugs, "index.md"],
  }));
}

export async function GET(_request: Request, { params }: MarkdownRouteContext) {
  const { slug } = await params;
  const pageSlug = slug?.at(-1) === "index.md" ? slug.slice(0, -1) : slug;
  const page = source.getPage(pageSlug, "en");

  if (!page) {
    return new Response("Not found", { status: 404 });
  }

  return new Response(await getLLMText(page), {
    headers: {
      "Content-Type": "text/markdown; charset=utf-8",
    },
  });
}
