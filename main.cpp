#include <iostream>
#include <gmp.h>
#include <cassert>
#include <gmpxx.h>
#include <string>

constexpr unsigned floor_log2(unsigned x) {
    return x==1 ? 0 : 1 + floor_log2(x >> 1);
}

class Mtgy {
    mpz_class N;
    mpz_class N_invert;
    mpz_class R;
    mpz_class R_invert;
    size_t n_size;

    void redc(mpz_class &r, const mpz_class &t) const {
        mpz_class m;
        mpz_mod_2exp(r.get_mpz_t(), t.get_mpz_t(), n_size);
        r *= N_invert;
        mpz_mod_2exp(r.get_mpz_t(), r.get_mpz_t(), n_size);

        r *= N;
        r += t;
        r >>= n_size;

        if (r >= N) {
            r -= N;
        }
    }

public:
    /**
     * @brief Construct a new Mtgy object
     * @param n should be odd
     */
    explicit Mtgy(mpz_class n) : N(std::move(n)), N_invert(), R(0), R_invert() {
        assert(N%2==1);
        this->n_size = mpz_sizeinbase(N.get_mpz_t(), 2);

        mpz_setbit(R.get_mpz_t(), n_size);
        mpz_invert(R_invert.get_mpz_t(), R.get_mpz_t(), N.get_mpz_t());

        // N' = (R * R' - 1) / N
        mpz_mul(N_invert.get_mpz_t(), R.get_mpz_t(), R_invert.get_mpz_t());
        N_invert -= 1;
        mpz_divexact(N_invert.get_mpz_t(), N_invert.get_mpz_t(), N.get_mpz_t());
    }

    /** Get into mtgy field */
    void into(mpz_class &r, const mpz_class &t) const {
        r = t*R%N;
    }

    /** Get out of mtgy field */
    void escape(mpz_class &r, const mpz_class &t) const {
        r = t*R_invert%N;
    }

    /**
     * `a` and `b` should be in mtgy field!
     * The result `r` will be in mtgy field!
     * @param r
     * @param a
     * @param b
     */
    void mul(mpz_class &r, const mpz_class &a, const mpz_class &b) const {
        mpz_class t = a*b;
        redc(r, t);
    }

    void debug() const {
        std::cout << "N = 0x" << N.get_str(16) << "; = " << N.get_str() << std::endl;
        std::cout << "R = 0x" << R.get_str(16) << "; = " << R.get_str() << std::endl;
        std::cout << "N' = 0x" << N_invert.get_str(16) << "; = " << N_invert.get_str()  << std::endl;
        std::cout << "R' = 0x" << R_invert.get_str(16) << "; = " << R_invert.get_str() << std::endl;
        std::cout << "n size = " << n_size << std::endl;
    }
};

/**
 * r = a^u % m. Do not actually use it!.
 * @param r
 * @param a
 * @param u
 * @param m
 */
void pow_mod_example(mpz_class &r, const mpz_class &a, const  mpz_class &u, const mpz_class &m) {

    Mtgy mtgy(m);
    mpz_class am, rm;
    mtgy.into(am, a);
    mtgy.into(rm ,1);

    // Just an example, don't do this.
    for(mpz_class i = 0; i < u; i++) {
        mtgy.mul(rm, rm, am);
    }

    mtgy.escape(r, rm);
}

/** Some test cases and example usage */
int main() {
    std::string cases[][4] = {
        {"1", "2", "13", "2"},
        {"1", "1", "13", "1"},
        {"7", "7", "13", "10"},
        {"2", "13", "207", "26"},
        {"1", "1", "1009", "1"},
        {"2", "10", "1009", "20"},
        {"5", "1", "193514046488575", "5"},
        {"15", "1", "4349330786055998253486590232462401", "15"},
        {"15", "10",
         "1475703270992002140168997557525132617116077748043980354291003276386587324053694848174953095546817655706234979251318204003655882580688895",
         "150"},
        {"148677972634832330983979593310074301486537017973460461278300587514468301043894574906886127642530475786889672304776052879927627556769456140664043088700743909632312483413393134504352834240399191134336344285483935856491230340093391784574980688823380828143810804684752914935441384845195613674104960646037368551517",
         "158741574437007245654463598139927898730476924736461654463975966787719309357536545869203069369466212089132653564188443272208127277664424448947476335413293018778018615899291704693105620242763173357203898195318179150836424196645745308205164116144020613415407736216097185962171301808761138424668335445923774195463",
         "446397596678771930935753654586920306936946621208913265356418844327220812727766442444894747633541329301877801861589929170469310562024276317335720389819531817915083642419664574530820516411614402061341540773621609718596217130180876113842466833544592377419546315874157443700724565446359813992789873047692473646165446397596678771930935753654586920306936946621208913265356418844327220812727766442444894747633541329301877801861589929170469310562045923774195463",
         "15733033542428556326610775226428250291950090984377467644096837926072"
         "98553857572965450727431838091748906310425930542328045644280094594289"
         "52380420588404540083723320848855612172087517363909606183916778041064"
         "11997952939978862543172484483575568826983703005515400230343351224994"
         "85403291437917132468481025327704901371719125205664144192914895118949"
         "25716605685210349843822514310138216212323303683754146084454361295646"
         "557462263542138176646203699553393662651092450"}
    };

    for (auto &c : cases) {
        mpz_class a (c[0]), b(c[1]), m(c[2]), r(c[3]);
        Mtgy mtgy(m);

        mpz_class am, bm, abm, ab;

        mtgy.into(am, a);
        mtgy.into(bm, b);

        mtgy.mul(abm, am, bm);

        mtgy.escape(ab, abm);

        assert(ab == r);
    }

    std::string pow_cases[][4] = {
        {"15", "117", "17", "2"},
        {"21251", "12415", "222221", "213559"}
    };
    for(auto &c: pow_cases) {
        mpz_class a (c[0]), u(c[1]), m(c[2]), r(c[3]);
        mpz_class res;
        pow_mod_example(res, a, u, m);
        assert(res == r);
    }
    return 0;
}
