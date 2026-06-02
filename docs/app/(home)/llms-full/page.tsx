import { getLLMSFullText } from "@/lib/llms-text";
import Link from "next/link";

export default async function LlmsFullPage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-6 px-6 py-12">
      <div>
        <p className="text-sm text-fd-muted-foreground">Agent-readable docs</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">llms-full</h1>
        <p className="mt-3 text-fd-muted-foreground">
          Browser preview for the canonical{" "}
          <Link className="underline underline-offset-4" href="/llms-full.txt">
            /llms-full.txt
          </Link>{" "}
          endpoint.
        </p>
      </div>
      <pre className="overflow-x-auto rounded-lg border bg-fd-muted p-4 text-sm leading-6">
        {await getLLMSFullText()}
      </pre>
    </main>
  );
}
