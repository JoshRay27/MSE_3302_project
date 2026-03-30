void setup() {
  // put your setup code here, to run once:

}

void loop() {
  // put your main code here, to run repeatedly:

}
#include <stdlib.h>  /* rand() */

/*
 * RPS outcome from player A's perspective.
 *   0 = paper, 1 = rock, 2 = scissors
 * Returns: +1 if a wins, 0 if tie, -1 if a loses.
 *
 * The key identity: a beats b iff (b - a + 3) % 3 == 1
 *   e.g. paper(0) beats rock(1):  (1-0+3)%3 = 1  ✓
 *        rock(1)  beats scissors(2): (2-1+3)%3 = 1  ✓
 *        scissors(2) beats paper(0): (0-2+3)%3 = 1  ✓
 */
static int rps_result(int a, int b) {
    if (a == b) return 0;
    return ((b - a + 3) % 3 == 1) ? 1 : -1;
}

/*
 * Decide which hand to remove.
 *
 *   my_right  – our right hand  (0=paper, 1=rock, 2=scissors)
 *   my_left   – our left hand
 *   opp_left  – opponent's left hand
 *   opp_right – opponent's right hand
 *
 * Returns 0  →  remove our RIGHT hand (play left)
 *         1  →  remove our LEFT  hand (play right)
 *
 * Requires rand() to be seeded (e.g. srand(analogRead(A0))) before first call.
 */
int choose_hand(int my_right, int my_left, int opp_left, int opp_right) {

    /* ── 2×2 payoff matrix (our perspective) ─────────────────────────── */
    int rr = rps_result(my_right, opp_right);  /* right vs their right */
    int rl = rps_result(my_right, opp_left);   /* right vs their left  */
    int lr = rps_result(my_left,  opp_right);  /* left  vs their right */
    int ll = rps_result(my_left,  opp_left);   /* left  vs their left  */

    /* ── Pure-strategy dominance check ───────────────────────────────── */
    /* Does our right hand weakly dominate our left? */
    if (rr >= lr && rl >= ll) return 1;  /* always play right */
    /* Does our left  hand weakly dominate our right? */
    if (lr >= rr && ll >= rl) return 0;  /* always play left  */

    /* ── Mixed Nash equilibrium ───────────────────────────────────────── *
     *                                                                     *
     * We solve for the prob p of playing right that makes the opponent    *
     * indifferent, which is also our own equilibrium mixing rate:         *
     *                                                                     *
     *   rr·p + rl·(1-p) = lr·p + ll·(1-p)                               *
     *   p = (ll - rl) / ((rr - lr) - (rl - ll))                          *
     *                                                                     *
     * Example – we play PR, opponent plays SR (lecture case):             *
     *   rr=-1, rl=1, lr=1, ll=0  →  p = (0-1)/((-1-1)-(1-0)) = -1/-3   *
     *   After normalisation: p = 1/3  →  play right (P) 1/3 of the time, *
     *                                    play left  (R) 2/3 of the time.  *
     *   R matches their R → play the matching hand 2/3. ✓                 *
     * ─────────────────────────────────────────────────────────────────── */
    int num   = ll - rl;
    int denom = (rr - lr) - (rl - ll);

    if (denom < 0) { num = -num; denom = -denom; }  /* ensure denom > 0 */
    if (denom == 0) return rand() & 1;               /* degenerate: coin flip */

    /* Play right with probability num/denom */
    return (rand() % denom < num) ? 1 : 0;
}