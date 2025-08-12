export const add: (a: number, b: number) => number;
export const load_module : (path: string) => void;
export const load_image  : (path: string) => void;
export const unload_module: () => void;
export const inference_start: (prompt: string,cbFn: (result: string) => void) => void;
export const inference_stop: () => void;