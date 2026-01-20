import { useEffect, useState, type ChangeEvent } from "react";

const apiBase = import.meta.env.VITE_API_BASE || "http://localhost:8000";

type ResultItem = {
  id: string;
  score?: number;
  text?: string;
  image_url?: string;
  url?: string;
  description?: string;
  metadata?: {
    keywords?: string[];
  };
  created_at?: string;
  width?: number;
  height?: number;
};

export default function App() {
  const [searchText, setSearchText] = useState("");
  const [searchImageFile, setSearchImageFile] = useState<File | null>(null);
  const [searchImagePreview, setSearchImagePreview] = useState<string | null>(
    null
  );
  const [items, setItems] = useState<ResultItem[]>([]);
  const [status, setStatus] = useState<string>("");
  const [isFiltered, setIsFiltered] = useState(false);

  const loadImages = async () => {
    setStatus("Loading latest images...");
    const response = await fetch(`${apiBase}/images?per_page=24`);
    const payload = await response.json();
    setItems(payload.items || []);
    setIsFiltered(false);
    setStatus(response.ok ? "Latest images loaded" : payload.detail || "error");
  };

  useEffect(() => {
    void loadImages();
  }, []);

  useEffect(() => {
    return () => {
      if (searchImagePreview) {
        URL.revokeObjectURL(searchImagePreview);
      }
    };
  }, [searchImagePreview]);

  const searchByText = async () => {
    if (!searchText.trim()) return;
    setStatus("Searching by text...");
    const response = await fetch(`${apiBase}/search/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: searchText, k: 10 }),
    });
    const payload = await response.json();
    setItems(payload.results || []);
    setIsFiltered(true);
    setStatus(response.ok ? "Similar images ready" : payload.detail || "error");
  };

  const searchByImage = async () => {
    if (!searchImageFile) return;
    setStatus("Searching by image...");
    const form = new FormData();
    form.append("file", searchImageFile);
    const response = await fetch(`${apiBase}/search/image?k=10`, {
      method: "POST",
      body: form,
    });
    const payload = await response.json();
    setItems(payload.results || []);
    setIsFiltered(true);
    setStatus(response.ok ? "Similar images ready" : payload.detail || "error");
  };

  const handleImageChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSearchImageFile(file);
    if (!file) {
      setSearchImagePreview(null);
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    setSearchImagePreview(previewUrl);
  };

  const buildCaption = (item: ResultItem) => {
    const description = item.description?.trim();
    if (description) {
      if (description.length <= 60) return description;
      return `${description.slice(0, 57)}...`;
    }
    const keywords = item.metadata?.keywords ?? [];
    const topKeywords = keywords.slice(0, 3);
    return topKeywords.length > 0 ? topKeywords.join(", ") : null;
  };

  return (
    <main>
      <header>
        <div>
          <h1>Image Similarity Explorer</h1>
          <p>
            Browse the latest images, then query by text or an image to find
            similar visuals.
          </p>
        </div>
        <div className="status">
          <span>{status}</span>
          {isFiltered && (
            <button className="ghost" onClick={loadImages}>
              Show latest
            </button>
          )}
        </div>
      </header>

      <section className="search">
        <div>
          <h2>Search by text</h2>
          <label>Describe what you want to see</label>
          <input
            type="text"
            value={searchText}
            onChange={(event) => setSearchText(event.target.value)}
            placeholder="E.g. misty forest, neon city, warm sunset"
          />
          <button onClick={searchByText}>Find similar images</button>
        </div>
        <div>
          <h2>Search by image</h2>
          <label>Upload a reference image</label>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
          />
          {searchImagePreview ? (
            <div className="search-preview">
              <img src={searchImagePreview} alt="Uploaded reference" />
              <span className="small">Reference image preview</span>
            </div>
          ) : (
            <span className="small">No reference image selected.</span>
          )}
          <button className="secondary" onClick={searchByImage}>
            Find similar images
          </button>
        </div>
      </section>

      <section>
        <h2>{isFiltered ? "Similar results" : "Latest images"}</h2>
        {items.length === 0 ? (
          <div className="empty">No images to show yet.</div>
        ) : (
          <div className="image-grid">
            {items.map((item) => {
              const imageSrc = item.image_url || item.url;
              const caption = buildCaption(item);
              const altText = item.description || caption || item.id;
              return (
                <article key={item.id} className="image-card">
                  {imageSrc ? (
                    <img src={imageSrc} alt={altText} />
                  ) : (
                    <div className="image-fallback">No image</div>
                  )}
                  <div className="image-meta">
                    {caption && (
                      <p className="caption">{caption}</p>
                    )}
                    <div className="meta-row">
                      {item.score !== undefined && (
                        <span>Score {item.score.toFixed(4)}</span>
                      )}
                      {item.width && item.height && (
                        <span>
                          {item.width}×{item.height}
                        </span>
                      )}
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>
    </main>
  );
}
