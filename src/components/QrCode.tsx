import QRCode from "qrcode";
import { useEffect, useState } from "react";

export function QrCode({ value, size = 220 }: { value: string; size?: number }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    QRCode.toDataURL(value, { margin: 1, width: size * 2, errorCorrectionLevel: "M" }).then(setSrc).catch(() => setSrc(null));
  }, [value, size]);
  return (
    <div className="qr-box">
      {src ? <img src={src} alt={`QR code for ${value}`} style={{ width: size, height: size }} /> : <div style={{ width: size, height: size }} />}
    </div>
  );
}
