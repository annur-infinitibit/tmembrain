/**
 * Shared FFI helpers for Membrain Node.js bindings.
 */

import koffi from "koffi";
import * as path from "path";
import * as os from "os";
import * as fs from "fs";

/** Error thrown by Membrain operations. */
export class MembrainError extends Error {
  constructor(
    message: string,
    public readonly code: number = -1
  ) {
    super(message);
    this.name = "MembrainError";
  }
}

export const MEMBRAIN_OK = 0;

export function findLibrary(): string {
  const envPath = process.env.MEMBRAIN_LIB_PATH;
  if (envPath) return envPath;

  const system = os.platform();
  let libName: string;
  if (system === "linux") {
    libName = "libmembrain_ffi.so";
  } else if (system === "darwin") {
    libName = "libmembrain_ffi.dylib";
  } else if (system === "win32") {
    libName = "membrain_ffi.dll";
  } else {
    libName = "libmembrain_ffi.so";
  }

  // Check common development locations
  const searchDirs = [
    __dirname,
    path.resolve(__dirname, ".."),
    path.resolve(__dirname, "..", "..", "target", "debug"),
    path.resolve(__dirname, "..", "..", "target", "release"),
  ];

  for (const dir of searchDirs) {
    const candidate = path.join(dir, libName);
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return libName;
}

// koffi type aliases
export const charPtr = koffi.pointer("char");
export const charPtrPtr = koffi.pointer(charPtr);
export const int64Ptr = koffi.pointer("int64_t");
export const voidPtr = koffi.pointer("void");
export const voidPtrPtr = koffi.pointer(voidPtr);

/**
 * Allocate and read an int64 output parameter from an FFI call.
 *
 * Usage:
 *   const out = newInt64Out();
 *   fn(handle, out);
 *   return readInt64(out);
 */
export function newInt64Out(): BigInt64Array {
  return new BigInt64Array(1);
}

export function readInt64(out: BigInt64Array): number {
  return Number(out[0]);
}

/**
 * Allocate a buffer for a string output parameter (char**) from an FFI call.
 *
 * Usage:
 *   const buf = newStringOut();
 *   fn(handle, query, k, buf);
 *   const { value, ptr } = readStringOut(buf);
 *   try { return JSON.parse(value); } finally { if (ptr) freeFn(ptr); }
 */
export function newStringOut(): any {
  return koffi.alloc("void*", 1);
}

export function readStringOut(buf: any): { value: string; ptr: any } {
  const rawPtr = koffi.decode(buf, "void*");
  if (!rawPtr) return { value: "", ptr: null };
  const value = koffi.decode(rawPtr, "char", -1) as string;
  return { value, ptr: rawPtr };
}
