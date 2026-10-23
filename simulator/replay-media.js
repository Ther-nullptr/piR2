/* One bounded decoder per precision. Forward seeks retain the last presented frame. */
"use strict";
window.ReplayMedia = (() => {
  const presenters = new WeakMap();
  class Presenter {
    constructor(card) {
      this.stage = card.querySelector(".media-stage");
      this.message = this.stage.querySelector(".media-message");
      this.caption = this.stage.querySelector(".frame-meta");
      this.media = null;
      this.presented = null;
      this.busy = false;
    }
    releaseMedia() {
      const media = this.media;
      this.media = null;
      this.presented = null;
      this.busy = false;
      if (!media) return;
      if (media.tagName === "VIDEO") media.pause?.();
      media.removeAttribute?.("src");
      if (media.tagName === "VIDEO") media.load?.();
      media.remove();
    }
    dispose() {
      this.releaseMedia();
      this.context = null;
      this.target = null;
    }
    update(asset, target, caption, context, epoch) {
      this.target = target;
      this.context = context;
      this.fallbackCaption = caption;
      if (!asset?.media_id || target == null) {
        this.releaseMedia();
        this.message.hidden = false;
        this.message.textContent = !asset?.media_id
          ? "该视角未记录媒体"
          : "游标之前没有帧";
        this.caption.textContent = caption;
        return;
      }
      const key = epoch + ":" + asset.media_id;
      if (!this.media || this.media.dataset.key !== key)
        this.create(asset, key);
      const video = this.media;
      if (video.tagName === "VIDEO") {
        // A backward scrub must not expose a frame from the old, later cursor.
        if (
          this.presented &&
          context &&
          this.presented.relative_s > context.cursor
        ) {
          video.classList.add("pending");
          this.message.hidden = false;
          this.message.textContent = "定位游标之前的实际帧…";
        }
        this.describe();
        this.pump();
      } else this.caption.textContent = caption;
    }
    create(asset, key) {
      this.releaseMedia();
      const media = document.createElement(
        asset.mime?.startsWith("video") ? "video" : "img",
      );
      this.media = media;
      media.dataset.key = key;
      media.dataset.asset = asset.media_id;
      media.classList.add("pending");
      this.message.hidden = false;
      this.message.textContent = "加载该运行的实际帧…";
      const current = () => this.media === media && media.isConnected;
      if (media.tagName === "VIDEO") {
        media.muted = true;
        media.playsInline = true;
        media.preload = "auto";
        media.addEventListener("loadeddata", () => {
          if (!current()) return;
          this.present();
          this.pump();
        });
        media.addEventListener("seeked", () => {
          if (!current()) return;
          this.busy = false;
          // Present a completed decode before admitting the newest queued target.
          this.present();
          requestAnimationFrame(() => {
            if (current()) this.pump();
          });
        });
      } else {
        media.alt = "固定输入 · 静态帧";
        media.addEventListener("load", () => {
          if (current()) {
            media.classList.remove("pending");
            this.message.hidden = true;
          }
        });
      }
      media.addEventListener("error", () => {
        if (!current()) return;
        media.classList.add("pending");
        this.message.hidden = false;
        this.message.textContent = "媒体无法解码或文件缺失 · " + asset.media_id;
      });
      this.stage.insertBefore(media, this.stage.firstChild);
      media.src = "/api/replay/media/" + encodeURIComponent(asset.media_id);
    }
    describe() {
      if (!this.presented) return;
      if (this.context && this.presented.relative_s > this.context.cursor) {
        this.caption.textContent = "等待游标之前的实际帧";
        return;
      }
      this.caption.textContent = `${this.presented.frame_id} · ${this.context?.timeLabel ?? "采集"} ${this.presented.relative_s.toFixed(3)} s · ${this.context?.ended ? "末帧保持" : "本精度独立轨迹"}`;
    }
    present() {
      const media = this.media;
      if (media.readyState < 2 || media.seeking) return;
      const frames = this.context?.frames ?? [];
      const frame = frames
        .filter((f) => f.media_time_s <= media.currentTime + 1e-6)
        .at(-1);
      if (!frame || frame.relative_s > this.context.cursor) {
        // A queued forward decode can finish after a backward cursor change.
        // Hide its newly drawable frame in this event, before the next RAF/seek.
        media.classList.add("pending");
        this.message.hidden = false;
        this.message.textContent = "等待游标之前的实际帧…";
        this.caption.textContent = "等待游标之前的实际帧";
        return;
      }
      this.presented = frame;
      media.dataset.presentedFrame = frame.frame_id;
      media.dataset.presentedWall = frame.relative_s;
      media.dataset.presentedMediaTime = frame.media_time_s;
      media.dataset.presentationCount =
        Number(media.dataset.presentationCount ?? 0) + 1;
      media.classList.remove("pending");
      this.message.hidden = true;
      this.describe();
    }
    pump() {
      const media = this.media;
      if (
        !media ||
        media.tagName !== "VIDEO" ||
        this.busy ||
        media.seeking ||
        media.readyState < 2
      )
        return;
      if (Math.abs(media.currentTime - this.target) <= 0.002) {
        this.present();
        return;
      }
      this.busy = true;
      // Do not blank a valid older frame on every forward decode.
      if (!this.presented || this.context?.cursor < this.presented.relative_s)
        media.classList.add("pending");
      media.currentTime = this.target;
    }
  }
  function update(card, asset, target, caption, context, epoch) {
    let presenter = presenters.get(card);
    if (!presenter) {
      presenter = new Presenter(card);
      presenters.set(card, presenter);
    }
    presenter.update(asset, target, caption, context, epoch);
  }
  function dispose(card) {
    presenters.get(card)?.dispose();
    presenters.delete(card);
  }
  return { update, dispose };
})();
