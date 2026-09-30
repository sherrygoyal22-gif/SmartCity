/* SmartCity UI behaviours that must work even if the map library fails to load. */
/* =========================
   PROFILE DROPDOWN + ADMIN SIDEBAR
========================= */
(() => {
    "use strict";

    const dropdowns = document.querySelectorAll("[data-dropdown]");

    const closeAll = (except) => {
        dropdowns.forEach((dropdown) => {
            if (dropdown !== except) {
                dropdown.classList.remove("open");
                dropdown.querySelector("[data-dropdown-toggle]")?.setAttribute("aria-expanded", "false");
            }
        });
    };

    dropdowns.forEach((dropdown) => {
        const toggle = dropdown.querySelector("[data-dropdown-toggle]");
        toggle?.addEventListener("click", (event) => {
            event.stopPropagation();
            const willOpen = !dropdown.classList.contains("open");
            closeAll(dropdown);
            dropdown.classList.toggle("open", willOpen);
            toggle.setAttribute("aria-expanded", String(willOpen));
        });
    });

    document.addEventListener("click", (event) => {
        if (!event.target.closest("[data-dropdown]")) closeAll();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeAll();
    });

    // Admin sidebar (mobile): the hamburger toggles it; tapping the dim backdrop closes it.
    const adminShell = document.querySelector(".admin-shell");
    document.querySelector("[data-admin-menu]")?.addEventListener("click", () => {
        adminShell?.classList.toggle("nav-open");
    });
    document.querySelector(".admin-backdrop")?.addEventListener("click", () => {
        adminShell?.classList.remove("nav-open");
    });

    // Show/hide password buttons on the login and register forms.
    document.querySelectorAll("[data-toggle-password]").forEach((button) => {
        button.addEventListener("click", () => {
            const input = document.getElementById(button.dataset.togglePassword);
            if (!input) return;
            const show = input.type === "password";
            input.type = show ? "text" : "password";
            button.setAttribute("aria-label", show ? "Hide password" : "Show password");
            button.classList.toggle("on", show);
        });
    });
})();

/* auto-dismiss flash messages (errors stay a little longer) */
document.querySelectorAll(".toast").forEach((toast) => {
    const delay = toast.classList.contains("error") ? 9000 : 4500;
    setTimeout(() => {
        toast.classList.add("hide");
        setTimeout(() => toast.remove(), 450);
    }, delay);
});

/* =========================
   INSTALLABLE APP (PWA): service worker + "Add to Home Screen" prompt
   Works on Android (Chrome/Edge/Samsung), iPhone/iPad (Safari) and desktop.
========================= */
(() => {
    "use strict";
    if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
        window.addEventListener("load", () => navigator.serviceWorker.register("/sw.js", {scope: "/"}).catch(() => {}));
    }

    const standalone = window.matchMedia("(display-mode: standalone)").matches || window.navigator.standalone === true;
    if (standalone) return;
    let dismissed = false;
    try { dismissed = Number(localStorage.getItem("sc_install_dismissed") || 0) > Date.now(); } catch (e) { /* storage blocked */ }
    if (dismissed) return;

    const ua = navigator.userAgent || "";
    const isIOS = /iphone|ipad|ipod/i.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
    const isMobile = isIOS || /android/i.test(ua);
    let deferred = null;

    const build = (message, withButton) => {
        const banner = document.createElement("div");
        banner.className = "install-banner show";
        banner.setAttribute("role", "dialog");
        banner.innerHTML = '<img src="/static/icons/icon-192.png" alt=""><div class="ib-text"><b>Install SmartCity</b><span></span></div>' +
            (withButton ? '<button class="ib-go" type="button">Install</button>' : "") + '<button class="ib-x" type="button" aria-label="Close">×</button>';
        banner.querySelector("span").textContent = message;
        const close = () => {
            banner.remove();
            try { localStorage.setItem("sc_install_dismissed", String(Date.now() + 7 * 864e5)); } catch (e) { /* ignore */ }
        };
        banner.querySelector(".ib-x").addEventListener("click", close);
        banner.querySelector(".ib-go")?.addEventListener("click", async () => {
            if (!deferred) return;
            deferred.prompt();
            await deferred.userChoice.catch(() => {});
            deferred = null;
            banner.remove();
        });
        document.body.appendChild(banner);
    };

    window.addEventListener("beforeinstallprompt", (event) => {
        event.preventDefault();
        deferred = event;
        if (isMobile) build("Add it to your home screen and open it like an app.", true);
    });
    if (isIOS) {
        // iOS Safari has no install event: show the manual steps once.
        window.addEventListener("load", () => setTimeout(() => {
            if (!document.querySelector(".install-banner")) build("Tap the Share button, then “Add to Home Screen”.", false);
        }, 2500));
    }
})();
