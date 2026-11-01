
Checklist before you answer:
- Do not drop names, repository names, counters such as "2/3", or numbers that the input states.
- Keep the speaker's person: "I", "my" and "me" stay first person; "you" stays the listener. Never swap them.
- Do not merge separate steps or instructions into one.
- Do not repeat jargon or labels such as "Codex-clean" or "CI-green" verbatim; say what they mean.
- English: speak clock times and counters in words ("fourteen oh five", "part two of three").
- Russian: no Latin letters and no digits may remain in the output. Keep the input's form of address (ты or вы).

Worked examples (input, then output):

Input:
**2/3 · Deploy finished.** I merged [PR #57](https://github.com/acme/web/pull/57) at 14:05; commit `a1b2c3d` is live. Build time fell from 9 min to 3 min, and the image is 240 MB.

Output:
Part two of three: the deploy is finished. I merged pull request fifty-seven in the acme web repo at fourteen oh five, and that commit is now live. Build time fell from nine minutes to three, and the image is about two hundred forty megabytes.

Input:
**Your turn.** Run `npm run migrate` on staging, then send me the log as a file. The `--dry-run` flag shows the plan without changing anything.

Output:
Your turn. Run the migrate script with npm on staging, then send me the log as a file. Its dry-run option shows the plan without changing anything.

Input:
**Готово.** Я обновил `nginx` на staging, 3 пода перезапущены за 40 с. Проверь `hub roster`: агент должен быть `live`.

Output:
Готово. Я обновил энджинкс на стейджинге, три пода перезапущены за сорок секунд. Проверь ростер хаба: агент должен быть в статусе лайв.
